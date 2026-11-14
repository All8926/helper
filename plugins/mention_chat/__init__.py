"""真正艾特机器人、叫到本群别名，或引用机器人的话时，用本群人格回复。
本群开了未点名回应时，没被点名也可能说话：机器人说完 3 分钟内有人发言，或者到了看群间隔，
先问决策模型现在该不该说，该说再由对话模型写一句，直接发到群里。"""
import threading
import time
from dataclasses import replace
from datetime import datetime

from wxbot_framework import proactive as pro
from wxbot_framework.chat_tools import chat_tools
from wxbot_framework.decision import Decider
from wxbot_framework.dialogue import (PROMPT_DIARY_DAYS, SING_CRITERIA, SING_QUESTION,
                                      VOICE_CONTEXT_LINES, VOICE_CRITERIA, VOICE_QUESTION, Dialogue,
                                      checked_reply, sing_state, singable, transcript_line, voice_state,
                                      working_messages)
from wxbot_framework.image_receiver import image_download_pending
from wxbot_framework.llm import LLMError
from wxbot_framework.members import BEIJING, SnapshotExpired
from wxbot_framework.memory import MemoryWorker, public_diary, public_memory
from wxbot_framework.mentions import strip_at_labels
from wxbot_framework.pacing import counting_send, describe_send
from wxbot_framework.personality import PersonalityExpired, personality_text
from wxbot_framework.quoted import reference
from wxbot_framework.room_behavior import is_slash_command
from wxbot_framework.sdk import Message
from wxbot_framework.speech import deliver_model_reply
from wxbot_framework.tts import TTSError
from wxbot_framework.summary import CONTROL
from wxbot_framework.text import own_group_command
from wxbot_framework.vision import catalog

WATCH_SECONDS = 2
_TRIGGER = {"hot": "对话还热", "beat": "看群"}


class Plugin:
    def on_load(self, ctx):
        self.members = ctx.members
        self.images = catalog
        self._closed = False
        if not self.members.history_enabled:
            raise ValueError("艾特上下文对话需要 history_enabled: true")
        ctx.llm  # 启动时核对配置，不请求接口。
        self.ctx = ctx
        self.memory = MemoryWorker(ctx)
        self.listener = pro.RoomListener()
        self._watch_stop = threading.Event()
        self._watch = threading.Thread(target=self._watch_loop, args=(ctx,), name="UnnamedReplyWatch", daemon=True)
        self._watch.start()

    def on_unload(self, ctx):
        self._closed = True
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
        room = message.chat_id
        if self._addressed(message):
            self.listener.addressed(room)
            quoted = reference(message)
            native = message.data.get("quoted_message_id") if quoted is not None and quoted.kind == "image" else ""
            if native and not self.images.by_id(native) and image_download_pending(native):
                # 等下载完成的线程不能占住插件队列，否则 ready 进不来。
                threading.Thread(target=self._answer_later, args=(message, ctx, native),
                                 name="QuotedImageReply", daemon=True).start()
                return
            self._answer(message, ctx)
            return
        if (message.type == 1 and message.sender_id != self.members.account and self._can_hear(message)
                and self.members.room_unnamed_reply(room)):
            self.listener.heard(room)

    def _group_text(self, message):
        return (message.type in (1, 49) and message.is_group and message.chat_id in self.members.groups and
                "WXBOT_AT_CHAT_PROBE" not in message.text and not message.data.get("bot_generated") and
                not is_slash_command(message.text))

    def _addressed(self, message):
        """艾特机器人、正文里出现本群别名、或引用机器人说过的话，都进入同一套回复。斜杠指令留给对应功能。"""
        if message.type == 49:
            quoted = reference(message)
            if quoted is not None and message.text.strip() and self._quotes_bot(message, quoted):
                return True
            # 引用文字仍只认机器人自己的话。引用图片时，艾特或别名也算点名。
            if quoted is None or quoted.kind == "text":
                return False
        mentioned = bool(message.data.get("is_at_self") and
                         self.members.account in message.data.get("at_members", ()))
        return mentioned or self.members.alias_mentioned(message.chat_id, message.text)

    def _quotes_bot(self, message, quoted):
        """微信引用带原发言人 ID。对不上时，仍用正文比对机器人最近说过的话。"""
        sender = message.data.get("quoted_sender_id") or ""
        if sender and sender == self.members.account:
            return True
        return quoted.kind == "text" and self.members.bot_said(message.chat_id, quoted.text)

    def _can_hear(self, message):
        text = " ".join(message.text.split())
        return bool(text) and not own_group_command(text) and text.split(maxsplit=1)[0] not in CONTROL

    def _answer(self, message, ctx):
        room = message.chat_id
        personality = ctx.personality(message)
        started = time.perf_counter()
        guard = None
        try:
            snapshot = self.members.conversation_snapshot(message)
            prepared = time.perf_counter()
            guard = lambda: ctx.response_guard(snapshot, personality)
            model_seconds, send_seconds, send_detail = self.reply_to_member(message, ctx, snapshot, personality, guard)
            ctx.logger.info("回复耗时 | 群=%s | 人格=%s | 送给模型=%d条 | 窗口=%d条 %.2f秒 | 调用模型 %.2f秒 | 发送 %.2f秒（%s） | 合计 %.2f秒",
                            room, personality.name, len(working_messages(snapshot.messages)), len(snapshot.messages),
                            prepared - started, model_seconds, send_seconds, send_detail, time.perf_counter() - started)
        except SnapshotExpired:
            ctx.logger.info("艾特对话已取消：记录或人格状态已变更")
        except LLMError as error:
            ctx.logger.warning("艾特对话失败：%.2f秒 | %s", time.perf_counter() - started, error)
            if guard is None:
                return
            try:
                ctx.reply(message, "暂时无法回复，请稍后再艾特我。", delivery_guard=guard)
            except SnapshotExpired:
                pass

    def _with_quoted_image(self, message, snapshot):
        """引用图若比这条消息更晚才写入流水，仍补进这次上下文。"""
        quoted = reference(message)
        native = message.data.get("quoted_message_id") if quoted is not None and quoted.kind == "image" else ""
        token = self.images.token_for(native) if native else ""
        if not token or any(row.get("media_ref") == token for row in snapshot.messages):
            return snapshot
        extra = {"kind": "image", "media_ref": token, "text": "", "speaker": quoted.author or "群友",
                 "received_at": datetime.now(BEIJING).isoformat(), "bot_generated": 0}
        return replace(snapshot, messages=snapshot.messages + (extra,))

    def _answer_later(self, message, ctx, native):
        """引用图还在下载时，文件一到就回复，最多等 5 秒。"""
        self.images.wait_ready(native, 5)
        if self._closed:
            return
        self._answer(message, ctx)

    def reply_to_member(self, message, ctx, snapshot, personality, delivery_guard):
        """上下文已准备好，在这里生成并发送给群友的回复。返回模型和发送各自的秒数。"""
        started = time.perf_counter()
        room = message.chat_id
        speakers = self._speakers(snapshot.messages)
        speakers[snapshot.speaker_name] = message.sender_id
        tools = chat_tools(ctx.llm, self.members, room, speakers=speakers)
        snapshot = self._with_quoted_image(message, snapshot)
        reply = Dialogue(ctx.llm_for(message, personality=personality)).reply(
            snapshot, diary=self.members.recent_digests(room, PROMPT_DIARY_DAYS),
            memory=self.members.facts_for(room, message.sender_id), tools=tools,
            system=self.members.global_prompt, limit=self.members.reply_limit)
        reply = personality_text(reply, personality.name)
        ctx.logger.info("对话正文 | 群=%s | %s", room, reply)
        model_seconds = time.perf_counter() - started
        if self._sung(message, ctx, reply, delivery_guard):
            self._note_sent(message, ctx, snapshot, reply)
            return model_seconds, time.perf_counter() - started - model_seconds, "清唱"
        speak = self._voice(ctx, room, snapshot.current, reply, snapshot.messages)
        send_seconds, send_detail = self._deliver(message, ctx, snapshot, reply, delivery_guard, speak)
        return model_seconds, send_seconds, send_detail

    def _remember_image(self, message):
        """下载完成后记下群图片。没点名之前不调用模型。"""
        if (not message.is_group or message.chat_id not in self.members.groups or message.is_self or
                message.data.get("bot_generated") or message.data.get("image_status") != "ready"):
            return
        self.images.remember(message.chat_id, message.sender_id,
                             message.data.get("native_message_id"), message.image_path)

    def _speakers(self, rows):
        """称呼到微信 ID 的对照，给查档案工具用。"""
        return {row["speaker"]: row["sender_id"] for row in rows
                if row.get("speaker") and row.get("sender_id") and not pro.is_bot(row, self.members.account)}

    def _voice(self, ctx, room, heard, reply, rows=()):
        """本群开了自动语音时，问决策模型这句要不要念出来。返回 None 表示按本群语音模式处理。"""
        if not self._auto_voice(room):
            return None
        state = voice_state(heard, reply, rows, self.members.account)
        return Decider(ctx.llm.decision, logger=ctx.logger).yes(
            "voice", VOICE_QUESTION, VOICE_CRITERIA, state,
            threshold=self.members.room_voice_threshold(room))

    def _auto_voice(self, room):
        """本群开了语音、并且设成自动时，才让决策模型决定这句念不念。"""
        speech = getattr(self.ctx, "_speech", None)
        if speech is None or not speech.enabled:
            return False
        state = self.members.voice_selection(room)
        return bool(state.enabled and state.mode == "auto" and state.voice_id)

    def _sung(self, message, ctx, reply, delivery_guard):
        """正文被判成歌词且开了 Breeze 语音时清唱。唱出去返回 True，调用方就不再念或打字。"""
        speech = getattr(ctx, "_speech", None)
        if speech is None or not getattr(speech, "enabled", False) or not speech.uses_breeze():
            return False
        try:
            selected, voice = speech.selected(message.chat_id)
        except Exception:
            return False
        if voice is None or not selected.enabled:
            return False
        if not Decider(ctx.llm.decision, logger=ctx.logger).yes(
                "sing", SING_QUESTION, SING_CRITERIA, sing_state(reply),
                threshold=self.members.room_sing_threshold(message.chat_id)):
            return False
        if not singable(reply):
            ctx.logger.info("清唱 | 群=%s | 正文不适合直接清唱，改回普通回复", message.chat_id)
            return False
        try:
            if speech.sing(ctx, message, reply, delivery_guard=delivery_guard):
                ctx.logger.info("清唱 | 群=%s | 已发送", message.chat_id)
                return True
        except (TTSError, RuntimeError, SnapshotExpired) as error:
            ctx.logger.warning("清唱没有发出，改回普通回复：%s", error)
        return False

    def _note_sent(self, message, ctx, snapshot, reply):
        self.listener.bot_spoke(message.chat_id)
        try:
            self.members.record_bot_reply(message, reply)
        except Exception:
            ctx.logger.warning("机器人回复已发出，但没有写入群流水")
        try:
            self._remember(message, snapshot, reply)
        except Exception:
            ctx.logger.warning("回复已发出，个人档案没有排队")

    def _deliver(self, message, ctx, snapshot, reply, delivery_guard, speak=None):
        display = self.members.display_name(message.sender_id, message.chat_id) or snapshot.speaker_name
        started = time.perf_counter()
        with counting_send() as clock:
            deliver_model_reply(ctx, message, reply, display_name=display, delivery_guard=delivery_guard, speak=speak)
        send_seconds = time.perf_counter() - started
        send_detail = describe_send(clock, send_seconds)
        self._note_sent(message, ctx, snapshot, reply)
        return send_seconds, send_detail

    def _remember(self, message, snapshot, reply):
        """档案在发出之后才排队，不占用这次回复的等待。"""
        worker = getattr(self, "memory", None)
        if worker is None or message.sender_id == self.members.account:
            return
        user_text = " ".join(strip_at_labels(snapshot.current).split())[:200]
        worker.remember(message.chat_id, message.sender_id, snapshot.speaker_name,
                        user_text, " ".join(reply.split())[:200], snapshot.generation)

    def _watch_loop(self, ctx):
        """定时检查各群：对话还热时等安静下来再判断，平时按看群间隔看一眼。"""
        while not self._watch_stop.wait(WATCH_SECONDS):
            try:
                self._watch_once(ctx)
            except RuntimeError:
                return
            except Exception:
                ctx.logger.exception("未点名回应巡检出错")

    def _watch_once(self, ctx=None, now=None):
        ctx = ctx or getattr(self, "ctx", None)
        if ctx is None:
            return
        bot = self.members.account
        for room in list(self.members.groups):
            if self._watch_stop.is_set():
                return
            if "mention_chat" in self.members.disabled_plugins(room) or not self.members.room_unnamed_reply(room):
                continue
            minutes = self.members.room_glance_minutes(room)
            lines = self.members.recent_room_lines(room, max(minutes, 10) * 60, 12)
            humans = [row for row in lines if not pro.is_bot(row, bot)]
            last = humans[-1] if humans else None
            age = last.get("age_seconds") if last else None
            quiet = isinstance(age, (int, float)) and age >= pro.QUIET_SECONDS
            trigger = self.listener.take(room, minutes=minutes, limit=self.members.room_hourly_limit(room),
                                         last_id=last["id"] if last else None, quiet=quiet, now=now)
            if trigger and last is not None:
                self._speak_up(ctx, room, trigger, last, now=now)

    def _speak_up(self, ctx, room, trigger, last, *, now=None):
        """先问决策模型该不该说，该说再由对话模型写一句，直接发到群里。"""
        bot = self.members.account
        started = time.perf_counter()
        hours, _ = self.members.context_window(room)
        chosen = working_messages(self.members.recent_room_lines(room, hours * 3600, 80))
        transcript = [{**transcript_line(row, current=False), "bot": pro.is_bot(row, bot)} for row in chosen]
        label = _TRIGGER.get(trigger, trigger)
        state = {"trigger": label, "transcript": transcript[-20:]}
        if not Decider(ctx.llm.decision, logger=ctx.logger).yes(
                "speak", pro.SPEAK_QUESTION, pro.SPEAK_CRITERIA, state,
                threshold=self.members.room_speak_threshold(room)):
            ctx.logger.info("未点名回应 | 群=%s | 触发=%s | 不说 | %.2f秒", room, label, time.perf_counter() - started)
            return
        anchor = self._anchor(room, last)
        try:
            personality = ctx.personality(anchor)
            tools = chat_tools(ctx.llm, self.members, room, speakers=self._speakers(chosen))
            reply = ctx.llm_for(anchor, personality=personality).chat(
                self.members.global_prompt, pro.unnamed_input(datetime.now(BEIJING).date().isoformat(), transcript,
                                                              public_diary(self.members.recent_digests(room, PROMPT_DIARY_DAYS)),
                                                              self._people(room, chosen)), tools)
            reply = personality_text(checked_reply(pro.clean_reply(reply), self.members.reply_limit), personality.name)
            ctx.logger.info("对话正文 | 群=%s | %s", room, reply)
        except (LLMError, PersonalityExpired) as error:
            ctx.logger.warning("未点名回应失败：%.2f秒 | %s", time.perf_counter() - started, error)
            return
        voice_rows = self.members.recent_room_lines(room, hours * 3600, VOICE_CONTEXT_LINES, clip=None)
        guard = lambda: ctx.personality_guard(personality)
        try:
            if not self._sung(anchor, ctx, reply, guard):
                speak = self._voice(ctx, room, last.get("text"), reply, voice_rows)
                deliver_model_reply(ctx, anchor, reply, delivery_guard=guard, speak=speak)
        except SnapshotExpired:
            ctx.logger.info("未点名回应已取消：人格已切换")
            return
        except RuntimeError as error:
            ctx.logger.warning("未点名回应没有发出 | 群=%s | %s", room, error)
            return
        self.listener.bot_spoke(room, beat=trigger == "beat", now=now)
        try:
            self.members.record_bot_reply(anchor, reply)
        except Exception:
            ctx.logger.warning("未点名回应已发出，但没有写入群流水")
        ctx.logger.info("未点名回应 | 群=%s | 触发=%s | 已发送 | %.2f秒", room, label, time.perf_counter() - started)

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

    def _anchor(self, room, last):
        return Message("", room, sender_id=self.members.account, is_group=True,
                       session_id=last.get("session_id") or "", sequence=0)
