"""真正艾特机器人，或叫到本群配置的别名后，用本群人格回复。没被点名时，本群打开了才考虑主动插一句。"""
import threading
import time

from wxbot_framework.dialogue import Dialogue, working_messages
from wxbot_framework.llm import LLMError
from wxbot_framework.members import SnapshotExpired
from wxbot_framework.memory import MemoryWorker
from wxbot_framework.mentions import strip_at_labels
from wxbot_framework.personality import PersonalityExpired, personality_text
from wxbot_framework.proactive import (
    COLD_CONTEXT_SECONDS, COLD_MAX_SECONDS, COLD_MIN_SECONDS, CONTEXT_LIMIT, CONTEXT_SECONDS,
    DEFAULT_COOLDOWN, ProactiveGate, bot_spoke_recently, candidate_reason, clean_reply,
    judge_gap, judge_system, transcript,
)
from wxbot_framework.room_behavior import is_slash_command
from wxbot_framework.sdk import Message
from wxbot_framework.speech import deliver_model_reply
from wxbot_framework.summary import CONTROL
from wxbot_framework.text import own_group_command

_REASON = {"mediate": "拉架", "question": "提问", "bot_thread": "接话",
           "unanswered": "未接的问题", "join": "话题", "cold": "冷场"}


class Plugin:
    def on_load(self, ctx):
        self.members = ctx.members
        if not self.members.history_enabled:
            raise ValueError("艾特上下文对话需要 history_enabled: true")
        ctx.llm  # 启动时核对配置，不请求接口。
        self.ctx = ctx
        self.memory = MemoryWorker(ctx)
        self.gate = ProactiveGate()
        self._cold_stop = threading.Event()
        self._cold = threading.Thread(target=self._cold_loop, args=(ctx,), name="ProactiveCold", daemon=True)
        self._cold.start()

    def on_unload(self, ctx):
        worker = getattr(self, "memory", None)
        if worker is not None:
            worker.stop()
        stop = getattr(self, "_cold_stop", None)
        if stop is not None:
            stop.set()

    def on_message(self, message, ctx):
        if not self._addressed(message):
            self._consider(message, ctx)
            return
        personality = ctx.personality(message)
        started = time.perf_counter()
        try:
            snapshot = self.members.conversation_snapshot(message)
            prepared = time.perf_counter()
            guard = lambda: ctx.response_guard(snapshot, personality)
            model_seconds, send_seconds = self.reply_to_member(message, ctx, snapshot, personality, guard)
            ctx.logger.info("回复耗时 | 群=%s | 人格=%s | 送给模型=%d条 | 窗口=%d条 %.2f秒 | 调用模型 %.2f秒 | 发送含等待 %.2f秒 | 合计 %.2f秒",
                            message.chat_id, personality.name, len(working_messages(snapshot.messages)),
                            len(snapshot.messages), prepared - started, model_seconds, send_seconds,
                            time.perf_counter() - started)
        except SnapshotExpired:
            ctx.logger.info("艾特对话已取消：记录或人格状态已变更")
        except LLMError as error:
            ctx.logger.warning("艾特对话失败：%.2f秒 | %s", time.perf_counter() - started, error)
            try:
                ctx.reply(message, "暂时无法回复，请稍后再艾特我。", delivery_guard=guard)
            except SnapshotExpired:
                pass

    def _addressed(self, message):
        """艾特机器人，或正文里出现本群别名，都进入同一套回复。斜杠指令留给对应功能。"""
        if (message.type != 1 or not message.is_group or message.chat_id not in self.members.groups or
                "WXBOT_AT_CHAT_PROBE" in message.text or message.data.get("bot_generated") or
                is_slash_command(message.text)):
            return False
        mentioned = bool(message.data.get("is_at_self") and
                         self.members.account in message.data.get("at_members", ()))
        return mentioned or self.members.alias_mentioned(message.chat_id, message.text)

    def reply_to_member(self, message, ctx, snapshot, personality, delivery_guard):
        """上下文已准备好，在这里生成并发送给群友的回复。返回模型和发送各自的秒数。"""
        started = time.perf_counter()
        diary = self.members.recent_digests(message.chat_id)
        memory = self.members.facts_for(message.chat_id, message.sender_id)
        reply = personality_text(Dialogue(ctx.llm_for(message, personality=personality)).reply(
            snapshot, diary=diary, memory=memory), personality.name)
        model_seconds = time.perf_counter() - started
        display = self.members.display_name(message.sender_id, message.chat_id) or snapshot.speaker_name
        started = time.perf_counter()
        deliver_model_reply(ctx, message, reply, display_name=display, delivery_guard=delivery_guard)
        send_seconds = time.perf_counter() - started
        # 发出去之后才记。写库失败不影响已经发出的回复。
        try:
            self.members.record_bot_reply(message, reply)
        except Exception:
            ctx.logger.warning("机器人回复已发出，但没有写入群流水")
        try:
            self._remember(message, snapshot, reply)
        except Exception:
            ctx.logger.warning("回复已发出，个人档案没有排队")
        return model_seconds, send_seconds

    def _remember(self, message, snapshot, reply):
        """档案在发出之后才排队，不占用这次回复的等待。"""
        worker = getattr(self, "memory", None)
        if worker is None or message.sender_id == self.members.account:
            return
        user_text = " ".join(strip_at_labels(snapshot.current).split())[:200]
        worker.remember(message.chat_id, message.sender_id, snapshot.speaker_name,
                        user_text, " ".join(reply.split())[:200], snapshot.generation)

    def _consider(self, message, ctx):
        """没被点名时先本地筛选。不像该接的话，不会打到模型。"""
        if not self._can_interject(message):
            return
        setting = self.members.room_settings.get(message.chat_id) or {}
        if not setting.get("proactive_enabled"):
            return
        prior = self.members.recent_room_lines(message.chat_id, CONTEXT_SECONDS, CONTEXT_LIMIT)
        if bot_spoke_recently(prior, self.members.account):
            return
        current_id = self._current_id(prior, message)
        reason = candidate_reason(message.text, prior, current_id=current_id, bot_id=self.members.account)
        if not reason:
            return
        cooldown = setting.get("proactive_cooldown", DEFAULT_COOLDOWN)
        if type(cooldown) is not int:
            cooldown = DEFAULT_COOLDOWN
        if not self.gate.allow(message.chat_id, cooldown, judge_gap(reason)):
            return
        self._speak(message, ctx, prior, reason, message.text)

    def _current_id(self, lines, message):
        text = " ".join(message.text.split())[:200]
        for row in reversed(lines):
            if (row.get("sequence") == message.sequence and row.get("sender_id") == message.sender_id
                    and row.get("text") == text):
                return row.get("id")
        return None

    def _can_interject(self, message):
        if (message.type != 1 or not message.is_group or message.chat_id not in self.members.groups or
                message.data.get("bot_generated") or message.sender_id == self.members.account or
                is_slash_command(message.text) or "WXBOT_AT_CHAT_PROBE" in message.text):
            return False
        text = " ".join(message.text.split())
        return bool(text) and not own_group_command(text) and text.split(maxsplit=1)[0] not in CONTROL

    def _speak(self, message, ctx, prior, reason, current):
        started = time.perf_counter()
        try:
            personality = ctx.personality(message)
            result = ctx.llm_for(message, personality=personality).complete_json(
                judge_system(reason), transcript(prior, current))
            raw = result.get("reply") if isinstance(result, dict) else ""
            cleaned = clean_reply(raw)
            reply = personality_text(cleaned, personality.name) if cleaned else ""
        except (LLMError, PersonalityExpired) as error:
            ctx.logger.warning("主动插话失败：%.2f秒 | %s", time.perf_counter() - started, error)
            return
        elapsed = time.perf_counter() - started
        label = _REASON.get(reason, reason)
        if not reply:
            ctx.logger.info("主动插话 | 群=%s | 原因=%s | 不说 | 模型 %.2f秒", message.chat_id, label, elapsed)
            return
        if not ctx.reply(message, reply):
            ctx.logger.warning("主动插话没有发出 | 群=%s", message.chat_id)
            return
        try:
            self.members.record_bot_reply(message, reply)
        except Exception:
            ctx.logger.warning("主动插话已发出，但没有写入群流水")
        self.gate.mark(message.chat_id)
        ctx.logger.info("主动插话 | 群=%s | 原因=%s | 已发送 | 模型 %.2f秒", message.chat_id, label, elapsed)

    def _cold_loop(self, ctx):
        """群里刚才还在聊、又安静了一会儿时，单独看要不要接一句。"""
        while not self._cold_stop.wait(60):
            try:
                self._cold_once(ctx)
            except RuntimeError:
                return
            except Exception:
                ctx.logger.exception("主动插话巡检出错")

    def _cold_once(self, ctx=None):
        ctx = ctx or getattr(self, "ctx", None)
        if ctx is None:
            return
        for room in list(self.members.groups):
            if self._cold_stop.is_set():
                return
            self._cold_room(ctx, room)

    def _cold_room(self, ctx, room):
        setting = self.members.room_settings.get(room) or {}
        if not setting.get("proactive_enabled") or "mention_chat" in self.members.disabled_plugins(room):
            return
        lines = self.members.recent_room_lines(room, COLD_MAX_SECONDS, CONTEXT_LIMIT)
        if not lines or bot_spoke_recently(lines, self.members.account):
            return
        age = lines[-1].get("age_seconds")
        if not isinstance(age, (int, float)) or not COLD_MIN_SECONDS <= age <= COLD_MAX_SECONDS:
            return
        cooldown = setting.get("proactive_cooldown", DEFAULT_COOLDOWN)
        if type(cooldown) is not int:
            cooldown = DEFAULT_COOLDOWN
        if not self.gate.allow(room, cooldown, 60):
            return
        anchor = Message("", room, sender_id=self.members.account, is_group=True,
                         session_id=lines[-1].get("session_id") or "", sequence=0)
        earlier = self.members.recent_room_lines(room, COLD_CONTEXT_SECONDS, CONTEXT_LIMIT)
        self._speak(anchor, ctx, earlier or lines, "cold", "")
