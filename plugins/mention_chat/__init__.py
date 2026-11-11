"""真正艾特机器人，或叫到本群配置的别名后，用本群人格回复。"""
import time

from wxbot_framework.dialogue import Dialogue, working_messages
from wxbot_framework.llm import LLMError
from wxbot_framework.members import SnapshotExpired
from wxbot_framework.memory import MemoryWorker
from wxbot_framework.mentions import strip_at_labels
from wxbot_framework.personality import personality_text
from wxbot_framework.room_behavior import is_slash_command
from wxbot_framework.speech import deliver_model_reply


class Plugin:
    def on_load(self, ctx):
        self.members = ctx.members
        if not self.members.history_enabled:
            raise ValueError("艾特上下文对话需要 history_enabled: true")
        ctx.llm  # 启动时核对配置，不请求接口。
        self.memory = MemoryWorker(ctx)

    def on_unload(self, ctx):
        worker = getattr(self, "memory", None)
        if worker is not None:
            worker.stop()

    def on_message(self, message, ctx):
        if not self._addressed(message):
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
