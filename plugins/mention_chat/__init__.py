"""真正艾特机器人、叫到本群别名，或引用机器人的话时，用本群人格回复。
本群活跃度大于 0 时，没被点名也会认出接着跟机器人说的话；每句话先在本地算开口意愿分，
过了本群门槛才问模型。群里在聊时还会隔几分钟让模型看一眼，冷场时也会掂量要不要接一句。"""
import threading
import time
from datetime import datetime

from wxbot_framework import proactive as pro
from wxbot_framework.dialogue import Dialogue, transcript_line, working_messages
from wxbot_framework.llm import LLMError
from wxbot_framework.members import BEIJING, SnapshotExpired
from wxbot_framework.memory import MemoryWorker, public_diary, public_memory
from wxbot_framework.mentions import strip_at_labels
from wxbot_framework.personality import PersonalityExpired, personality_text
from wxbot_framework.quoted import reference
from wxbot_framework.room_behavior import is_slash_command
from wxbot_framework.sdk import Message
from wxbot_framework.speech import deliver_model_reply
from wxbot_framework.summary import CONTROL
from wxbot_framework.text import own_group_command
from wxbot_framework.vision import SeenImages, vision_data_url

_REASON = {"mediate": "拉架", "question": "提问", "unanswered": "未接的问题",
           "join": "话题", "cold": "冷场", "glance": "看群"}
WATCH_SECONDS = 30


class Plugin:
    def on_load(self, ctx):
        self.members = ctx.members
        self.images = SeenImages()
        if not self.members.history_enabled:
            raise ValueError("艾特上下文对话需要 history_enabled: true")
        ctx.llm  # 启动时核对配置，不请求接口。
        self.ctx = ctx
        self.memory = MemoryWorker(ctx)
        self.gate = pro.ProactiveGate()
        self.followups = pro.FollowUps()
        self._cold_seen = {}
        self._glanced = {}
        self._watch_stop = threading.Event()
        self._watch = threading.Thread(target=self._watch_loop, args=(ctx,), name="ProactiveWatch", daemon=True)
        self._watch.start()

    def on_unload(self, ctx):
        worker = getattr(self, "memory", None)
        if worker is not None:
            worker.stop()
        stop = getattr(self, "_watch_stop", None)
        if stop is not None:
            stop.set()

    def on_message(self, message, ctx):
        if message.type == 3:
            self._remember_image(message)
            return
        if not self._group_text(message):
            return
        room, bot = message.chat_id, self.members.account
        if self._addressed(message):
            if pro.annoyed(message.text, (), None, bot, named=True):
                self._mute(ctx, room)
            self._answer(message, ctx)
            return
        if message.type != 1 or message.sender_id == bot or not self._can_interject(message):
            return
        level = self.members.room_activity(room)
        if level == 0:
            return
        # 群里每多一句话，机器人就多攒一点发言份额。
        self.gate.earn(room, pro.target_share(level))
        self._settle(ctx, room)
        recent = self.members.recent_room_lines(room, pro.CANDIDATE_SECONDS, 20)
        current_id = self._current_id(recent, message)
        if pro.annoyed(message.text, recent, current_id, bot):
            self._mute(ctx, room)
            return
        if self.gate.muted(room):
            return
        if self._follows_up(message, recent, current_id):
            self._answer(message, ctx, follow_up=True)
            return
        self._consider(message, ctx, recent, current_id, level)

    def _group_text(self, message):
        return (message.type in (1, 49) and message.is_group and message.chat_id in self.members.groups and
                "WXBOT_AT_CHAT_PROBE" not in message.text and not message.data.get("bot_generated") and
                not is_slash_command(message.text))

    def _addressed(self, message):
        """艾特机器人、正文里出现本群别名、或引用机器人说过的话，都进入同一套回复。斜杠指令留给对应功能。"""
        if message.type == 49:
            quoted = reference(message)
            if (quoted is not None and quoted.kind == "text" and bool(message.text.strip()) and
                    self.members.bot_said(message.chat_id, quoted.text)):
                return True
            # 引用文字仍只认机器人自己的话。引用图片时，艾特或别名也算点名。
            if quoted is None or quoted.kind == "text":
                return False
        mentioned = bool(message.data.get("is_at_self") and
                         self.members.account in message.data.get("at_members", ()))
        return mentioned or self.members.alias_mentioned(message.chat_id, message.text)

    def _answer(self, message, ctx, *, follow_up=False):
        room, sender = message.chat_id, message.sender_id
        personality = ctx.personality(message)
        started = time.perf_counter()
        guard = None
        try:
            snapshot = self.members.conversation_snapshot(message)
            prepared = time.perf_counter()
            guard = lambda: ctx.response_guard(snapshot, personality)
            if follow_up:
                result = self._follow_up(message, ctx, snapshot, personality, guard)
                if result is None:
                    self.followups.close(room, sender)
                    ctx.logger.info("接话识别 | 群=%s | 不是在跟机器人说 | 模型 %.2f秒", room, time.perf_counter() - prepared)
                    return
                model_seconds, send_seconds = result
                self.followups.advance(room, sender)
            else:
                model_seconds, send_seconds = self.reply_to_member(message, ctx, snapshot, personality, guard)
                self.followups.open(room, sender)
            if self.gate.engaged(room):
                self._feedback(ctx, room, pro.ENGAGED_STEP, "有人接了插话")
            ctx.logger.info("回复耗时 | 群=%s | 方式=%s | 人格=%s | 送给模型=%d条 | 窗口=%d条 %.2f秒 | 调用模型 %.2f秒 | 发送含等待 %.2f秒 | 合计 %.2f秒",
                            room, "接话" if follow_up else "点名", personality.name,
                            len(working_messages(snapshot.messages)), len(snapshot.messages), prepared - started,
                            model_seconds, send_seconds, time.perf_counter() - started)
        except SnapshotExpired:
            ctx.logger.info("艾特对话已取消：记录或人格状态已变更")
        except LLMError as error:
            ctx.logger.warning("%s失败：%.2f秒 | %s", "接话识别" if follow_up else "艾特对话",
                               time.perf_counter() - started, error)
            if follow_up or guard is None:
                return
            try:
                ctx.reply(message, "暂时无法回复，请稍后再艾特我。", delivery_guard=guard)
            except SnapshotExpired:
                pass

    def reply_to_member(self, message, ctx, snapshot, personality, delivery_guard):
        """上下文已准备好，在这里生成并发送给群友的回复。返回模型和发送各自的秒数。"""
        started = time.perf_counter()
        diary = self.members.recent_digests(message.chat_id)
        memory = self.members.facts_for(message.chat_id, message.sender_id)
        client = ctx.llm_for(message, personality=personality)
        image = self._vision_image(message, ctx)
        voice = self._auto_voice(message.chat_id)
        try:
            reply, speak = Dialogue(client).reply(
                snapshot, diary=diary, memory=memory, choose_voice=voice, image=image)
        except LLMError as error:
            # 文本模型拒绝图片时，仍按原来的文字回复。
            if not image or getattr(error, "status", None) != 400:
                raise
            ctx.logger.info("看图 | 群=%s | 模型未接受图片，改为只看文字", message.chat_id)
            reply, speak = Dialogue(client).reply(
                snapshot, diary=diary, memory=memory, choose_voice=voice)
        reply = personality_text(reply, personality.name)
        model_seconds = time.perf_counter() - started
        return model_seconds, self._deliver(message, ctx, snapshot, reply, delivery_guard, speak)

    def _remember_image(self, message):
        """下载完成后记下群图片。没点名之前不调用模型。"""
        if (not message.is_group or message.chat_id not in self.members.groups or message.is_self or
                message.data.get("bot_generated") or message.data.get("image_status") != "ready"):
            return
        self.images.remember(message.chat_id, message.sender_id,
                             message.data.get("native_message_id"), message.image_path)

    def _vision_image(self, message, ctx):
        """点名时附上引用到的图；没有引用图时，用这个人三分钟内刚发的一张。"""
        quoted = reference(message)
        if quoted is not None and quoted.kind == "image":
            path = self.images.by_id(message.data.get("quoted_message_id"))
            if not path:
                ctx.logger.info("看图 | 群=%s | 引用的图片还没有本地文件", message.chat_id)
                return None
        else:
            path = self.images.latest(message.chat_id, message.sender_id)
            if not path:
                return None
        url = vision_data_url(path)
        if not url:
            ctx.logger.info("看图 | 群=%s | 图片未能读取，改为只看文字", message.chat_id)
            return None
        ctx.logger.info("看图 | 群=%s | 已附上图片", message.chat_id)
        return url

    def _follow_up(self, message, ctx, snapshot, personality, delivery_guard):
        """模型判断不是在跟机器人说话时返回 None。"""
        started = time.perf_counter()
        reply, speak = Dialogue(ctx.llm_for(message, personality=personality)).reply(
            snapshot, diary=self.members.recent_digests(message.chat_id),
            memory=self.members.facts_for(message.chat_id, message.sender_id), follow_up=True,
            choose_voice=self._auto_voice(message.chat_id))
        model_seconds = time.perf_counter() - started
        if not reply:
            return None
        reply = personality_text(reply, personality.name)
        return model_seconds, self._deliver(message, ctx, snapshot, reply, delivery_guard, speak)

    def _deliver(self, message, ctx, snapshot, reply, delivery_guard, speak=None):
        display = self.members.display_name(message.sender_id, message.chat_id) or snapshot.speaker_name
        started = time.perf_counter()
        deliver_model_reply(ctx, message, reply, display_name=display, delivery_guard=delivery_guard, speak=speak)
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
        return send_seconds

    def _remember(self, message, snapshot, reply):
        """档案在发出之后才排队，不占用这次回复的等待。"""
        worker = getattr(self, "memory", None)
        if worker is None or message.sender_id == self.members.account:
            return
        user_text = " ".join(strip_at_labels(snapshot.current).split())[:200]
        worker.remember(message.chat_id, message.sender_id, snapshot.speaker_name,
                        user_text, " ".join(reply.split())[:200], snapshot.generation)

    def _mute(self, ctx, room):
        if not self.gate.muted(room):
            ctx.logger.info("主动插话 | 群=%s | 有人嫌机器人吵，1 小时内不主动开口", room)
            self._feedback(ctx, room, pro.ANNOYED_STEP, "有人嫌吵")
        self.gate.mute(room)
        self.followups.close(room)

    def _settle(self, ctx, room, now=None):
        if self.gate.settle(room, now):
            self._feedback(ctx, room, pro.IGNORED_STEP, "上一句插话没人接")

    def _feedback(self, ctx, room, step, why):
        """群友的反应让本群门槛慢慢升降，写进库里，重启后保留。"""
        try:
            value = self.members.shift_proactive_bias(room, step, pro.BIAS_LOW, pro.BIAS_HIGH)
        except Exception:
            ctx.logger.warning("主动插话门槛修正没有保存 | 群=%s", room)
            return
        ctx.logger.info("主动插话 | 群=%s | %s，门槛修正 %+.2f", room, why, value)

    def _follows_up(self, message, recent, current_id):
        room, sender = message.chat_id, message.sender_id
        if self.followups.active(room, sender):
            return True
        if self.followups.exhausted(room, sender):
            return False
        return pro.addresses_bot(message.text, recent, current_id, self.members.account)

    def _auto_voice(self, room):
        """本群开了语音、并且设成自动时，才让模型决定这句念不念。"""
        speech = getattr(self.ctx, "_speech", None)
        if speech is None or not speech.enabled:
            return False
        state = self.members.voice_selection(room)
        return bool(state.enabled and state.mode == "auto" and state.voice_id)

    def _can_interject(self, message):
        text = " ".join(message.text.split())
        return bool(text) and not own_group_command(text) and text.split(maxsplit=1)[0] not in CONTROL

    @staticmethod
    def _current_id(lines, message):
        text = " ".join(message.text.split())[:200]
        for row in reversed(lines):
            if (row.get("sequence") == message.sequence and row.get("sender_id") == message.sender_id
                    and row.get("text") == text):
                return row.get("id")
        return None

    def _consider(self, message, ctx, recent, current_id, level):
        """意愿分没过本群门槛时不会打到模型。"""
        room = message.chat_id
        score, reason = pro.willingness(message.text, recent, current_id=current_id,
                                        bot_id=self.members.account, level=level)
        if not reason:
            return
        bar = pro.speak_bar(level, self.gate.tokens(room), self.members.proactive_bias(room))
        if score < bar or not self.gate.allow(room, pro.JUDGE_GAP):
            return
        self._interject(ctx, room, level, reason, message, current_id, basis=f"意愿={score:.2f} 门槛={bar:.2f}")

    def _interject(self, ctx, room, level, reason, target, current_id=None, *, basis="", now=None):
        """和点名回复看同一段上下文。模型说不必开口就沉默；判断期间有人先说了也放弃。"""
        bot = self.members.account
        started = time.perf_counter()
        hours, _ = self.members.context_window(room)
        lines = self.members.recent_room_lines(room, hours * 3600, 80)
        chosen = working_messages(lines)
        transcript = [transcript_line(row, current=row["id"] == current_id) for row in chosen]
        recent_bot = [row["text"] for row in chosen if pro.is_bot(row, bot)][-3:]
        last_id = lines[-1]["id"] if lines else None
        personality = ctx.personality(target)
        choose_voice = self._auto_voice(room)
        try:
            result = ctx.llm_for(target, personality=personality).complete_json(
                pro.judge_system(reason, level, choose_voice=choose_voice),
                pro.judge_input(datetime.now(BEIJING).date().isoformat(), transcript,
                                public_diary(self.members.recent_digests(room)), self._people(room, chosen),
                                recent_bot))
            cleaned = pro.clean_reply(result.get("reply") if isinstance(result, dict) else "")
            reply = personality_text(cleaned, personality.name) if cleaned else ""
        except (LLMError, PersonalityExpired) as error:
            ctx.logger.warning("主动插话失败：%.2f秒 | %s", time.perf_counter() - started, error)
            return
        elapsed = time.perf_counter() - started
        label = _REASON.get(reason, reason)
        if not reply:
            ctx.logger.info("主动插话 | 群=%s | 活跃度=%d | 原因=%s | %s | 不说 | 模型 %.2f秒",
                            room, level, label, basis, elapsed)
            return
        if self._moved_on(room, last_id):
            ctx.logger.info("主动插话 | 群=%s | 活跃度=%d | 原因=%s | %s | 有人先说了，放弃 | 模型 %.2f秒",
                            room, level, label, basis, elapsed)
            return
        # 主动插话一律直接发到群里。模型就算返回 quote，也不引用、不艾特。
        speak = result.get("voice") if choose_voice and isinstance(result, dict) else None
        if type(speak) is not bool:
            speak = None
        guard = lambda: ctx.personality_guard(personality)
        try:
            deliver_model_reply(ctx, target, reply, delivery_guard=guard, speak=speak)
        except SnapshotExpired:
            ctx.logger.info("主动插话已取消：人格已切换")
            return
        except RuntimeError as error:
            ctx.logger.warning("主动插话没有发出 | 群=%s | %s", room, error)
            return
        try:
            self.members.record_bot_reply(target, reply)
        except Exception:
            ctx.logger.warning("主动插话已发出，但没有写入群流水")
        self.gate.mark(room, now)
        ctx.logger.info("主动插话 | 群=%s | 活跃度=%d | 原因=%s | %s | 已发送 | 模型 %.2f秒",
                        room, level, label, basis, elapsed)

    def _people(self, room, rows):
        """窗口里最近说话的几位群友的档案。"""
        people, seen = [], set()
        for row in reversed(rows):
            sender = row.get("sender_id")
            if not sender or pro.is_bot(row, self.members.account) or sender in seen:
                continue
            seen.add(sender)
            facts = public_memory(self.members.facts_for(room, sender))
            if facts:
                people.append({"name": row.get("speaker") or "群友", "memory": facts})
            if len(seen) >= 4:
                break
        return people

    def _moved_on(self, room, last_id):
        if last_id is None:
            return False
        return any(row["id"] > last_id and not pro.is_bot(row, self.members.account)
                   for row in self.members.recent_room_lines(room, 600, 10))

    def _watch_loop(self, ctx):
        """定时巡检各群：冷场时掂量接一句，群里在聊时隔几分钟让模型看一眼。"""
        while not self._watch_stop.wait(WATCH_SECONDS):
            try:
                self._watch_once(ctx)
            except RuntimeError:
                return
            except Exception:
                ctx.logger.exception("主动插话巡检出错")

    def _watch_once(self, ctx=None, now=None):
        ctx = ctx or getattr(self, "ctx", None)
        if ctx is None:
            return
        for room in list(self.members.groups):
            if self._watch_stop.is_set():
                return
            if "mention_chat" in self.members.disabled_plugins(room):
                continue
            level = self.members.room_activity(room)
            if level == 0:
                continue
            self._settle(ctx, room, now)
            if self.gate.muted(room, now):
                continue
            lines = self.members.recent_room_lines(room, pro.COLD_MAX_SECONDS, 12)
            if lines:
                self._cold_room(ctx, room, level, lines, now)
                self._glance_room(ctx, room, level, lines, now)

    def _cold_room(self, ctx, room, level, lines, now=None):
        bot = self.members.account
        if pro.bot_spoke_recently(lines, bot):
            return
        last = lines[-1]
        age = last.get("age_seconds")
        # 同一段安静只掂量一次。
        if (not isinstance(age, (int, float)) or not pro.COLD_MIN_SECONDS <= age <= pro.COLD_MAX_SECONDS or
                self._cold_seen.get(room) == last["id"]):
            return
        self._cold_seen[room] = last["id"]
        score = pro.cold_score(lines, bot, level)
        bar = pro.speak_bar(level, self.gate.tokens(room), self.members.proactive_bias(room))
        if score < bar or not self.gate.allow(room, pro.COLD_JUDGE_GAP, now):
            return
        self._interject(ctx, room, level, "cold", self._anchor(room, last),
                        basis=f"意愿={score:.2f} 门槛={bar:.2f}", now=now)

    def _glance_room(self, ctx, room, level, lines, now=None):
        """群里还在聊、份额也够一句时，隔一段时间让模型看一眼，由它判断现在适不适合插话。"""
        minutes = self.members.room_glance_minutes(room)
        tokens = self.gate.tokens(room)
        if minutes == 0 or tokens < 1:
            return
        moment = time.monotonic() if now is None else now
        last_time, last_id = self._glanced.get(room, (None, None))
        interval = pro.glance_interval(minutes, self.members.proactive_bias(room))
        if last_time is not None and moment - last_time < interval:
            return
        if not pro.glance_due(lines, self.members.account, last_id) or not self.gate.allow(room, pro.JUDGE_GAP, now):
            return
        self._glanced[room] = (moment, lines[-1]["id"])
        self._interject(ctx, room, level, "glance", self._anchor(room, lines[-1]),
                        basis=f"份额={tokens:.2f}", now=now)

    def _anchor(self, room, last):
        return Message("", room, sender_id=self.members.account, is_group=True,
                       session_id=last.get("session_id") or "", sequence=0)
