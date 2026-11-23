"""名单里的联系人发来私聊文字时，按这个人自己的人格回复。语音、音色和深度思考也按人分开。"""
import threading
import time
from dataclasses import replace
from datetime import datetime

from wxbot_framework.chat_tools import private_chat_tools
from wxbot_framework.decision import Decider
from wxbot_framework.dialogue import (SING_CRITERIA, SING_QUESTION, VOICE_CRITERIA, VOICE_QUESTION,
                                      Dialogue, quote_hint, quoted_image_token, sing_state, singable, voice_state)
from wxbot_framework.image_receiver import image_download_pending
from wxbot_framework.llm import LLMError
from wxbot_framework.members import BEIJING, SnapshotExpired
from wxbot_framework.personality import personality_text
from wxbot_framework.quoted import reference
from wxbot_framework.room_behavior import is_slash_command
from wxbot_framework.tts import TTSError
from wxbot_framework.vision import catalog


class Plugin:
    def on_load(self, ctx):
        self.members = ctx.members
        self.images = catalog
        self._closed = False
        if not self.members.history_enabled:
            raise ValueError("私聊对话需要 history_enabled: true")
        ctx.llm  # 启动时核对配置，不请求接口。

    def on_unload(self, ctx):
        self._closed = True

    def on_message(self, message, ctx):
        if message.type == 3:
            self._remember_image(message)
            return
        if not self._accepted(message):
            return
        native = self._pending_quote(message)
        if native:
            # 等下载的线程不能占住插件队列。
            threading.Thread(target=self._answer_later, args=(message, ctx, native),
                             name="PrivateQuotedImage", daemon=True).start()
            return
        self._answer(message, ctx)

    def _accepted(self, message):
        """只回复名单里的个人文字。群、自己发出的消息和斜杠指令都不进对话。"""
        if message.is_group or str(message.chat_id).endswith("@chatroom"):
            return False
        if message.chat_id not in self.members.private_contacts:
            return False
        if (message.is_self or message.sender_id == self.members.account or
                message.data.get("bot_generated")):
            return False
        if message.type == 1:
            text = message.text.strip()
            return bool(text) and not is_slash_command(text)
        if message.type == 49 and message.data.get("app_type") == 57:
            return bool(str(message.text).strip()) and not is_slash_command(message.text)
        return False

    def _pending_quote(self, message):
        quoted = reference(message) if message.type == 49 else None
        native = message.data.get("quoted_message_id") if quoted is not None and quoted.kind == "image" else ""
        if native and not self.images.by_id(native) and image_download_pending(native):
            return native
        return ""

    def _answer_later(self, message, ctx, native):
        """引用图还在下载时，文件一到就回复，最多等 5 秒。"""
        self.images.wait_ready(native, 5)
        if self._closed:
            return
        self._answer(message, ctx)

    def _answer(self, message, ctx):
        if _over_limit(ctx):
            self._reply_asleep(message, ctx)
            return
        started = time.perf_counter()
        guard = None
        try:
            personality = self.members.private_personality(message.chat_id)
            snapshot = self.members.conversation_snapshot(message)
            guard = lambda: ctx.response_guard(snapshot, personality)
            self.reply_to_friend(message, ctx, snapshot, personality, guard)
            ctx.logger.info("私聊回复耗时 | 会话=%s | %.2f秒", message.chat_id, time.perf_counter() - started)
        except SnapshotExpired:
            ctx.logger.info("私聊回复已取消：记录或人格已变更")
        except (LLMError, RuntimeError) as error:
            ctx.logger.warning("私聊回复失败：%.2f秒 | %s", time.perf_counter() - started, error)
            if guard is None:
                return
            try:
                ctx.reply(message, "暂时无法回复，请稍后再试。", delivery_guard=guard)
            except SnapshotExpired:
                pass

    def _reply_asleep(self, message, ctx):
        """到了每日上限就不再叫模型，直接回这一句。"""
        from wxbot_framework.token_usage import SLEEP_REPLY
        try:
            personality = self.members.private_personality(message.chat_id)
            snapshot = self.members.conversation_snapshot(message)
            guard = lambda: ctx.response_guard(snapshot, personality)
            if not ctx.reply(message, SLEEP_REPLY, delivery_guard=guard):
                ctx.logger.warning("上限提示没有发出 | 私聊=%s", message.chat_id)
                return
            try:
                self.members.record_bot_reply(message, SLEEP_REPLY)
            except Exception:
                ctx.logger.warning("上限提示已发出，但没有写入流水")
            ctx.logger.info("今日对话已达上限 | 私聊=%s", message.chat_id)
        except SnapshotExpired:
            ctx.logger.info("私聊回复已取消：记录或人格已变更")

    def reply_to_friend(self, message, ctx, snapshot, personality, delivery_guard):
        """上下文已准备好，按这个人的语音设置发送回复。"""
        tools = private_chat_tools(ctx.llm, self.members, message.chat_id)
        snapshot = self._with_quoted_image(message, snapshot)
        reply = Dialogue(ctx.llm_for(message, personality=personality)).reply(
            snapshot, diary=(), memory=(), tools=tools,
            system=self.members.global_prompt, limit=self.members.reply_limit,
            quote=quote_hint(message), quoted_image=quoted_image_token(message, self.images))
        reply = personality_text(reply, personality.name)
        ctx.logger.info("对话正文 | 私聊=%s | %s", message.chat_id, reply)
        self._send(message, ctx, snapshot, reply, delivery_guard)
        try:
            self.members.record_bot_reply(message, reply)
        except Exception:
            ctx.logger.warning("私聊回复已发出，但没有写入流水")
        return reply

    def _send(self, message, ctx, snapshot, reply, delivery_guard):
        """先尝试这个人的清唱和语音，没有发出就发纯文字。不艾特。"""
        if self._sung(message, ctx, reply, delivery_guard):
            return
        if self._voiced(message, ctx, snapshot, reply, delivery_guard):
            return
        if not ctx.reply(message, reply, delivery_guard=delivery_guard):
            raise RuntimeError("私聊回复没有发出")

    def _speech(self, ctx):
        speech = getattr(ctx, "_speech", None)
        if getattr(speech, "enabled", False) is not True:
            return None
        return speech

    def _sung(self, message, ctx, reply, delivery_guard):
        person = self.members.private_person(message.chat_id) or {}
        speech = self._speech(ctx)
        if person.get("voice_mode") == "off" or speech is None or not speech.can_sing():
            return False
        try:
            selected, voice = speech.selected(message.chat_id)
        except Exception:
            return False
        if voice is None or not selected.enabled:
            return False
        if not Decider(ctx.llm.decision, logger=ctx.logger).yes(
                "sing", SING_QUESTION, SING_CRITERIA, sing_state(reply),
                threshold=person["sing_threshold"]):
            return False
        if not singable(reply):
            return False
        try:
            return bool(speech.sing(ctx, message, reply, delivery_guard=delivery_guard))
        except (TTSError, RuntimeError, SnapshotExpired) as error:
            ctx.logger.warning("清唱没有发出，改回普通回复：%s", error)
            if isinstance(error, SnapshotExpired):
                raise
            return False

    def _voiced(self, message, ctx, snapshot, reply, delivery_guard):
        person = self.members.private_person(message.chat_id) or {}
        speech = self._speech(ctx)
        if person.get("voice_mode", "off") == "off" or speech is None:
            return False
        speak = None
        if person["voice_mode"] == "auto":
            state = voice_state(snapshot.current, reply, snapshot.messages, self.members.account)
            speak = Decider(ctx.llm.decision, logger=ctx.logger).yes(
                "voice", VOICE_QUESTION, VOICE_CRITERIA, state, threshold=person["voice_threshold"])
        try:
            return bool(speech.reply(ctx, message, reply, delivery_guard=delivery_guard, speak=speak))
        except SnapshotExpired:
            raise
        except TTSError as error:
            ctx.logger.warning("私聊语音没有发出，改回文字：%s", error)
            return False

    def _with_quoted_image(self, message, snapshot):
        """引用图若还没写进流水，仍补进这次上下文。"""
        quoted = reference(message)
        native = message.data.get("quoted_message_id") if quoted is not None and quoted.kind == "image" else ""
        token = self.images.token_for(native) if native else ""
        if not token or any(row.get("media_ref") == token for row in snapshot.messages):
            return snapshot
        extra = {"kind": "image", "media_ref": token, "text": "", "speaker": quoted.author or "对方",
                 "received_at": datetime.now(BEIJING).isoformat(), "bot_generated": 0}
        return replace(snapshot, messages=snapshot.messages + (extra,))

    def _remember_image(self, message):
        """下载完成后记下这张私聊图片。图片本身不触发回复。"""
        if (message.is_group or message.chat_id not in self.members.private_contacts or message.is_self or
                message.data.get("bot_generated") or message.data.get("image_status") != "ready"):
            return
        self.images.remember(message.chat_id, message.sender_id,
                             message.data.get("native_message_id"), message.image_path)


def _over_limit(ctx):
    from wxbot_framework.token_usage import dialogue_blocked
    try:
        return dialogue_blocked(ctx.llm)
    except Exception:
        return False
