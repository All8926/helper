"""未点名回应：对话还热时等安静再判断，平时按看群间隔看一眼；该不该说由决策模型定，说什么由对话模型写。"""
import io
import json
import threading
import time
import unittest
from datetime import timedelta
from unittest.mock import patch

from PIL import Image

import test_chat_summary as daily
from fake_llm import FakeLLM
from plugins.mention_chat import Plugin
from wxbot_framework import proactive as pro
from wxbot_framework.chat_tools import chat_tools
from wxbot_framework.dialogue import SING_CRITERIA, sing_state, singable
from wxbot_framework.llm import LLMError
from wxbot_framework.room_behavior import BehaviorError, normalize_global_prompt, normalize_reply_limit, normalize_room_setting
from wxbot_framework.sdk import Message
from wxbot_framework.vision import catalog, vision_data_url

ROOM, CAT, ROBOT, OTHER = daily.ROOM, daily.CAT, daily.ROBOT, daily.OTHER


class FilterTests(unittest.TestCase):
    def test_global_prompt_is_empty_without_a_length_cap(self):
        self.assertEqual(normalize_global_prompt(None), "")
        self.assertEqual(normalize_global_prompt("  "), "")
        long = "词" * 5000
        self.assertEqual(normalize_global_prompt(long), long)
        with self.assertRaises(BehaviorError):
            normalize_global_prompt("坏\0词")

    def test_settings_default_and_validate(self):
        default = normalize_room_setting({}, [])
        self.assertFalse(default["unnamed_reply"])
        self.assertEqual((default["glance_minutes"], default["hourly_limit"], default["voice_threshold"]), (2, 3, 0.6))
        self.assertNotIn("voice_rate", default)
        self.assertEqual(normalize_room_setting({"voice_rate": 9}, [])["voice_threshold"], 0.4)
        self.assertEqual(normalize_room_setting({"voice_threshold": 0.7, "voice_rate": 1}, [])["voice_threshold"], 0.7)
        self.assertEqual(default["speak_threshold"], 0.5)
        self.assertEqual(default["sing_threshold"], 0.85)
        self.assertEqual(normalize_room_setting({"sing_threshold": 0.4}, [])["sing_threshold"], 0.4)
        self.assertEqual(normalize_room_setting({"speak_threshold": 0.8}, [])["speak_threshold"], 0.8)
        self.assertEqual(normalize_room_setting({"speak_threshold": 0}, [])["speak_threshold"], 0.0)
        self.assertNotIn("activity", default)
        self.assertTrue(normalize_room_setting({"activity": 4}, [])["unnamed_reply"])
        self.assertFalse(normalize_room_setting({"activity": 0}, [])["unnamed_reply"])
        self.assertTrue(normalize_room_setting({"proactive_enabled": True}, [])["unnamed_reply"])
        self.assertFalse(normalize_room_setting({"unnamed_reply": False, "activity": 9}, [])["unnamed_reply"])
        self.assertEqual(normalize_room_setting({"glance_minutes": 0, "hourly_limit": 0}, [])["hourly_limit"], 0)
        self.assertEqual(normalize_room_setting({"voice_threshold": 0.25}, [])["voice_threshold"], 0.25)
        self.assertTrue(default["deep_thinking"])
        self.assertFalse(normalize_room_setting({"deep_thinking": False}, [])["deep_thinking"])
        self.assertEqual(normalize_reply_limit(None), 600)
        self.assertEqual(normalize_reply_limit(8000), 8000)
        for bad in (0, 8001, True, 1.5, "600"):
            with self.assertRaises(BehaviorError):
                normalize_reply_limit(bad)
        for bad in ({"unnamed_reply": "yes"}, {"glance_minutes": 61}, {"hourly_limit": 31}, {"hourly_limit": True},
                    {"speak_threshold": 1.1}, {"speak_threshold": True},
                    {"voice_threshold": -0.1}, {"voice_threshold": True},
                    {"sing_threshold": 1.2}, {"sing_threshold": True}, {"deep_thinking": "yes"}):
            with self.subTest(bad=bad), self.assertRaises(BehaviorError):
                normalize_room_setting(bad, [])

    def test_hot_window_waits_for_quiet_and_mentions_cancel_it(self):
        listener = pro.RoomListener()
        listener.heard(ROOM, now=0)
        self.assertIsNone(listener.take(ROOM, minutes=0, limit=3, last_id=1, quiet=True, now=10))
        listener.bot_spoke(ROOM, now=100)
        listener.heard(ROOM, now=150)
        self.assertIsNone(listener.take(ROOM, minutes=0, limit=3, last_id=2, quiet=False, now=153))
        self.assertEqual(listener.take(ROOM, minutes=0, limit=0, last_id=2, quiet=False, now=157), "hot")
        self.assertIsNone(listener.take(ROOM, minutes=0, limit=3, last_id=2, quiet=True, now=160))
        listener.heard(ROOM, now=200)
        listener.addressed(ROOM)
        self.assertIsNone(listener.take(ROOM, minutes=0, limit=3, last_id=3, quiet=True, now=210))
        listener.heard(ROOM, now=100 + pro.HOT_SECONDS + 1)
        self.assertIsNone(listener.take(ROOM, minutes=0, limit=3, last_id=4, quiet=True, now=400))

    def test_beat_needs_interval_new_quiet_lines_and_room_in_the_hourly_limit(self):
        listener = pro.RoomListener()
        take = lambda now, last_id=1, quiet=True, limit=1, minutes=2: listener.take(
            ROOM, minutes=minutes, limit=limit, last_id=last_id, quiet=quiet, now=now)
        self.assertIsNone(take(0))
        self.assertIsNone(take(119, last_id=2))
        self.assertIsNone(take(121))
        self.assertIsNone(take(121, last_id=2, quiet=False))
        self.assertIsNone(take(121, last_id=None))
        self.assertIsNone(take(121, last_id=2, minutes=0))
        self.assertEqual(take(121, last_id=2), "beat")
        listener.bot_spoke(ROOM, beat=True, now=121)
        listener.bot_spoke(ROOM, now=200)
        self.assertIsNone(take(300, last_id=3))
        self.assertEqual(take(300, last_id=3, limit=2), "beat")
        self.assertEqual(take(121 + pro.HOUR, last_id=4), "beat")

    def test_reply_cleanup(self):
        self.assertEqual(pro.clean_reply("  试试\n先重启 "), "试试 先重启")
        self.assertEqual(pro.clean_reply(None), "")


class PluginTests(unittest.TestCase):
    setUp = daily.DailySummaryTests.setUp
    new_store = daily.DailySummaryTests.new_store
    message = daily.DailySummaryTests.message
    model = daily.DailySummaryTests.model

    def tearDown(self):
        plugin = getattr(self, "plugin", None)
        if plugin is not None:
            plugin.on_unload(self.ctx)
        daily.DailySummaryTests.tearDown(self)

    def start(self, unnamed=True, speak=1.0, decision=None, **setting):
        # 夹具默认停在北京时间午夜前一秒，往后拨时钟会触发每日清理。
        self.now = self.now.replace(hour=4)
        self.store.room_settings[ROOM] = {"unnamed_reply": unnamed, **setting}
        self.backend.capabilities = ("text.quote.send",)
        self.backend.send_quote_text = lambda room, original_id, text: self.backend.sent.append((room, text, original_id)) or True
        self.backend.send_at_text = lambda room, text, ids: self.backend.sent.append((room, text, ids)) or True
        catalog.clear()
        self.unnamed = "要不吃烧烤"
        self.named, self.spoken = [], []

        def chat(system, user, tools=(), **_kwargs):
            body = json.loads(user)
            if "people" in body:
                self.spoken.append((system, body, [tool.name for tool in tools]))
                return self.unnamed
            self.named.append((system, user))
            return "这是回答。"

        self.ctx._llm = FakeLLM(chat=chat, decision=decision or {"fallback": {"speak": speak}})
        self.plugin = Plugin()
        self.plugin.on_load(self.ctx)

    def say(self, text, sequence, sender=CAT, **data):
        msg = self.message(text, sender=sender, sequence=sequence, data={"native_message_id": sequence, **data})
        self.store.record(msg)
        self.plugin.on_message(msg, self.ctx)
        return msg

    def at(self, text, sequence, sender=CAT):
        return self.say("@机器人\u2005" + text, sequence, sender=sender, at_members=(ROBOT,), is_at_self=True)

    def later(self, seconds=pro.QUIET_SECONDS + 1):
        return time.monotonic() + seconds

    def test_custom_global_prompt_is_the_chat_task(self):
        self.start(unnamed=False)
        self.store.global_prompt = "请用短句"
        self.at("在吗", 1)
        self.assertIn("请用短句", self.named[-1][0])
        self.assertNotIn("禁止聊政治", self.named[-1][0])

    def test_lyric_reply_is_sung_without_the_tag(self):
        self.start(unnamed=False, decision={"fallback": {"sing": 1}})

        class Speech:
            enabled = True
            sung = []

            def can_sing(self):
                return True

            def selected(self, room):
                from types import SimpleNamespace
                return SimpleNamespace(enabled=True), SimpleNamespace(id="voc_xeh3w54cqvnp")

            def sing(self, ctx, message, text, delivery_guard=None):
                self.sung.append(text)
                return True

        speech = Speech()
        self.ctx._speech = speech
        self.at("唱一句", 1)
        self.assertEqual(speech.sung, ["这是回答。"])
        self.assertFalse(any("[唱歌]" in str(item) for item in self.backend.sent))

    def test_sing_decision_reads_only_the_full_reply(self):
        lyric = "想把夜空唱给你听" * 6
        self.start(unnamed=False, decision={"enabled": True, "api_key": "decision-key"})

        def chat(system, user, tools=(), **_kwargs):
            return lyric

        self.ctx._llm.chat.side_effect = chat

        class Speech:
            enabled = True
            sung = []

            def can_sing(self):
                return True

            def selected(self, room):
                from types import SimpleNamespace
                return SimpleNamespace(enabled=True), SimpleNamespace(id="voc_xeh3w54cqvnp")

            def sing(self, ctx, message, text, delivery_guard=None):
                self.sung.append(text)
                return True

        self.ctx._speech = Speech()
        answer = io.BytesIO(json.dumps({"answers": {"sing": {"noul": 0.9}}}).encode())
        with patch("wxbot_framework.llm.urlopen", return_value=answer) as call:
            self.at("前面聊了很久的歌", 1)
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["questions"]["sing"]["criteria"]["true"], SING_CRITERIA[0])
        self.assertEqual(body["state"], sing_state(lyric))
        self.assertNotIn("transcript", body["state"])
        self.assertGreater(len(lyric), 40)
        self.assertTrue(singable(lyric))
        self.assertFalse(singable("看这个 https://example.com/a"))
        self.assertEqual(self.ctx._speech.sung, [lyric])

    def test_unnamed_reply_off_only_answers_mentions(self):
        self.start(unnamed=False)
        self.at("今晚吃什么", 1)
        self.say("那明天吃啥", 2)
        self.plugin._watch_once(now=self.later())
        self.plugin._watch_once(now=self.later(3600))
        self.assertEqual(self.spoken, [])
        self.assertEqual(self.backend.sent, [(ROOM, "这是回答。", 1)])

    def test_hot_conversation_gets_one_plain_line_after_quiet(self):
        self.start()
        self.store.replace_facts(ROOM, CAT, ("喜欢火锅",), 0)
        self.at("今晚吃什么", 1)
        self.say("那明天吃啥", 2)
        self.plugin._watch_once(now=self.later(1))
        self.assertEqual(self.spoken, [])
        self.plugin._watch_once(now=self.later())
        self.assertEqual(self.backend.sent[-1], (ROOM, "要不吃烧烤"))
        system, body, tools = self.spoken[0]
        self.assertNotIn("禁止聊政治", system)
        self.assertEqual([row["text"][-5:] for row in body["transcript"]], ["今晚吃什么", "这是回答。", "那明天吃啥"])
        self.assertEqual([row["bot"] for row in body["transcript"]], [False, True, False])
        self.assertEqual(body["people"][0]["memory"], ["喜欢火锅"])
        self.assertIn("search_history", tools)
        self.assertIn("view_image", tools)
        self.assertNotIn("web_search", tools)
        self.assertFalse(any(row.get("type") == "image" and row.get("id") for row in body["transcript"]))
        self.assertEqual(self.store.recent_room_lines(ROOM, 60, 5)[-1]["text"], "要不吃烧烤")
        self.plugin._watch_once(now=self.later(10))
        self.assertEqual(len(self.spoken), 1)

    def test_decision_no_keeps_the_bot_quiet(self):
        self.start(speak=0.0)
        self.at("今晚吃什么", 1)
        self.say("那明天吃啥", 2)
        self.plugin._watch_once(now=self.later())
        self.assertEqual(self.spoken, [])
        self.assertEqual(len(self.backend.sent), 1)

    def test_room_speak_threshold_decides_the_same_probability(self):
        self.start(speak_threshold=0.8, decision={"enabled": True, "api_key": "decision-key"})
        self.at("今晚吃什么", 1)
        self.say("那明天吃啥", 2)
        answer = io.BytesIO(json.dumps({"answers": {"speak": {"noul": 0.6}}}).encode())
        with patch("wxbot_framework.llm.urlopen", return_value=answer):
            self.plugin._watch_once(now=self.later())
        self.assertEqual(self.spoken, [])
        self.store.room_settings[ROOM]["speak_threshold"] = 0.5
        self.say("那你再说一次", 3)
        again = io.BytesIO(json.dumps({"answers": {"speak": {"noul": 0.6}}}).encode())
        with patch("wxbot_framework.llm.urlopen", return_value=again):
            self.plugin._watch_once(now=self.later(20))
        self.assertEqual(len(self.spoken), 1)

    def test_mention_cancels_the_pending_hot_check(self):
        self.start()
        self.at("今晚吃什么", 1)
        self.say("那明天吃啥", 2)
        self.at("你说呢", 3)
        self.plugin._watch_once(now=self.later())
        self.assertEqual(self.spoken, [])
        self.assertEqual(len(self.named), 2)

    def test_decision_model_sees_the_trigger_and_chat(self):
        self.start(decision={"enabled": True, "api_key": "decision-key"})
        self.at("今晚吃什么", 1)
        self.say("那明天吃啥", 2)
        answer = io.BytesIO(json.dumps({"answers": {"speak": {"noul": 0.2}}}).encode())
        with patch("wxbot_framework.llm.urlopen", return_value=answer) as call:
            self.plugin._watch_once(now=self.later())
        state = json.loads(call.call_args.args[0].data)["state"]
        self.assertEqual(state["trigger"], "对话还热")
        self.assertEqual(state["transcript"][-1]["text"], "那明天吃啥")
        self.assertEqual(self.spoken, [])

    def test_beat_runs_on_its_interval_and_respects_the_hourly_limit(self):
        self.start(glance_minutes=2, hourly_limit=1)
        self.unnamed = "我推荐香山"

        def line(text, sequence, sender=CAT):
            self.store.record(self.message(text, sender=sender, sequence=sequence))
            self.now += timedelta(seconds=10)

        line("我们周末去爬山吧", 1)
        self.plugin._watch_once(now=1000)
        line("好啊去哪座山", 2, sender=OTHER)
        self.plugin._watch_once(now=1100)
        self.assertEqual(self.spoken, [])
        self.plugin._watch_once(now=1121)
        self.assertEqual(self.backend.sent, [(ROOM, "我推荐香山")])
        self.assertEqual(self.spoken[0][1]["transcript"][-1]["text"], "好啊去哪座山")
        line("香山人多吗", 3)
        self.plugin._watch_once(now=1300)
        self.assertEqual(len(self.spoken), 1)
        line("那早点出发", 4)
        self.plugin._watch_once(now=1121 + pro.HOUR)
        self.assertEqual(len(self.spoken), 2)

    def test_beat_waits_for_quiet_and_is_off_at_zero_minutes(self):
        self.start(glance_minutes=2)
        self.store.record(self.message("我们周末去爬山吧", sequence=1))
        self.plugin._watch_once(now=1000)
        self.store.record(self.message("好啊去哪座山", sender=OTHER, sequence=2))
        self.plugin._watch_once(now=1200)
        self.assertEqual(self.spoken, [])
        self.now += timedelta(seconds=10)
        self.store.room_settings[ROOM]["glance_minutes"] = 0
        self.plugin._watch_once(now=1400)
        self.assertEqual(self.spoken, [])
        self.store.room_settings[ROOM]["glance_minutes"] = 2
        self.plugin._watch_once(now=1500)
        self.assertEqual(len(self.spoken), 1)

    def test_failed_or_blank_unnamed_reply_sends_nothing(self):
        self.start(glance_minutes=1)
        self.store.record(self.message("我们周末去爬山吧", sequence=1))
        self.now += timedelta(seconds=10)
        self.plugin._watch_once(now=1000)
        replies = iter(("  ", LLMError("模拟超时")))

        def chat(*_args, **_kwargs):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        self.ctx._llm.chat.side_effect = chat
        for sequence in (2, 3):
            self.store.record(self.message(f"第{sequence}句", sequence=sequence))
            self.now += timedelta(seconds=10)
            self.plugin._watch_once(now=1000 + sequence * 100)
            self.assertEqual(self.ctx._llm.chat.call_count, sequence - 1)
        self.assertEqual(self.backend.sent, [])

    def test_lookup_tools_read_this_room_only(self):
        self.start(unnamed=False)
        self.store.replace_facts(ROOM, CAT, ("喜欢火锅",), 0)
        for sequence, (text, room) in enumerate((("周末去爬山", ROOM), ("100%_爬山", ROOM), ("别的群爬山", daily.OTHER_ROOM),
                                                  ("今天吃面", ROOM)), start=1):
            self.store.record(self.message(text, room=room, sequence=sequence))
        with self.store._connection() as db:
            db.execute("INSERT INTO room_digests(account,room,local_day,summary) VALUES(?,?,?,?)",
                       (ROBOT, ROOM, "2026-10-09", "大家约了  周末爬山"))
        tools = {tool.name: tool for tool in chat_tools(self.ctx.llm, self.store, ROOM, speakers={"小猫": CAT})}
        self.assertEqual(json.loads(tools["lookup_memory"].run({"name": "小猫"})), ["喜欢火锅"])
        self.assertEqual(json.loads(tools["lookup_memory"].run({"name": "测试成员"})), ["喜欢火锅"])
        self.assertIn("没有关于", tools["lookup_memory"].run({"name": "别人"}))
        self.assertIn("群里没有", tools["lookup_memory"].run({"name": "路人甲"}))
        self.assertEqual(json.loads(tools["lookup_diary"].run({"days": 99})),
                         [{"date": "2026-10-09", "summary": "大家约了 周末爬山"}])
        found = tools["search_history"].run({"keyword": "爬山", "days": 3})
        self.assertEqual([line.split("：", 1)[1] for line in found.splitlines()], ["周末去爬山", "100%_爬山"])
        self.assertEqual(tools["search_history"].run({"keyword": "%_"}).split("：", 1)[1], "100%_爬山")
        self.assertEqual(tools["search_history"].run({"keyword": "火车"}), "没有找到相关聊天")

    def test_quoting_a_bot_line_gets_a_reply(self):
        self.start(unnamed=False)
        self.at("今晚吃什么", 1)
        quoted = Message("真的假的", ROOM, sender_id=OTHER, is_group=True, session_id="test", sequence=2, type=49,
                         data={"app_type": 57, "quoted_message": {"author": "机器人", "kind": "text", "text": "这是回答。",
                                                                  "available": True, "truncated": False}})
        self.store.record(quoted)
        self.plugin.on_message(quoted, self.ctx)
        # 引用消息本身不能再被原生引用，回复改为艾特对方。
        self.assertEqual(self.backend.sent[-1][2], (OTHER,))
        self.assertTrue(self.backend.sent[-1][1].endswith("这是回答。"))
        self.assertEqual(len(self.backend.sent), 2)

    def test_quoting_the_bot_by_sender_replies_even_when_the_original_is_voice(self):
        self.start(unnamed=False)
        voice = Message("你刚才说的啥", ROOM, sender_id=OTHER, is_group=True, session_id="test", sequence=1, type=49,
                        data={"app_type": 57, "quoted_sender_id": ROBOT,
                              "quoted_message": {"author": "机器人", "kind": "voice", "text": "[语音]",
                                                 "available": True, "truncated": False}})
        self.store.record(voice)
        self.plugin.on_message(voice, self.ctx)
        self.assertEqual(len(self.backend.sent), 1)
        someone = Message("你刚才说的啥", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=2, type=49,
                          data={"app_type": 57, "quoted_sender_id": OTHER,
                                "quoted_message": {"author": "别人", "kind": "voice", "text": "[语音]",
                                                   "available": True, "truncated": False}})
        self.store.record(someone)
        self.plugin.on_message(someone, self.ctx)
        self.assertEqual(len(self.backend.sent), 1)

    def test_model_error_gets_a_short_apology(self):
        self.start(unnamed=False)
        self.ctx._llm.chat.side_effect = LLMError("大模型接口 HTTP 401；请核对密钥、余额和模型配置", status=401)
        self.at("在吗", 1)
        self.assertIn("暂时无法回复", self.backend.sent[-1][1])

    def _png(self, name, color):
        path = self.root / name
        Image.new("RGB", (4, 4), color).save(path)
        return path

    def _picture(self, path, sender=CAT, identifier="88", status="ready", sequence=1):
        message = Message("", ROOM, sender_id=sender, is_group=True, session_id="test",
            sequence=sequence, type=3, data={"image_status": status, "image_path": str(path),
                                             "native_message_id": identifier})
        self.store.record(message)
        self.plugin.on_message(message, self.ctx)
        return message

    def _look(self):
        """记下上下文里的图片编号。模型自己决定看时，测试替它看最后一张。"""
        self.offered, self.looked, self.systems, self.payloads = [], [], [], []

        def chat(system, user, tools=(), image=None, **_kwargs):
            if image is not None:
                self.looked.append(image)
                return "一张纯色图片"
            found = {tool.name: tool for tool in tools}
            body = json.loads(user)
            self.payloads.append(body)
            ids = [row["id"] for row in body.get("transcript", []) if row.get("type") == "image" and row.get("id")]
            self.offered.append(ids)
            self.systems.append(system)
            self.tools = found
            if "view_image" in found and ids:
                return found["view_image"].run({"id": ids[-1], "question": "这是什么"})
            return "看到了。"

        self.ctx._llm.chat.side_effect = chat

    def test_image_lines_carry_ids_and_paths_are_rejected(self):
        self.start(unnamed=False)
        self._look()
        red, blue = self._png("red.png", "red"), self._png("blue.png", "blue")
        self._picture(blue, sender=OTHER, identifier="81", sequence=1)
        self.at("看别人的图", 2)
        self.assertTrue(self.offered[-1])
        self.assertNotIn("附带了一张图片", self.systems[-1])
        self._picture(red, status="failed", identifier="82", sequence=3)
        self.at("失败的图", 4)
        failed = [row for row in self.payloads[-1]["transcript"] if row.get("type") == "image" and "id" not in row]
        self.assertTrue(failed)
        self._picture(red, identifier="83", sequence=5)
        self.at("这是什么", 6)
        self.assertIn(vision_data_url(red), self.looked)
        self.assertEqual(self.tools["view_image"].run({"id": r"C:\secret.png"}), "没有这张图片")
        self.assertEqual(self.tools["view_image"].run({"id": "404"}), "没有这张图片")

    def test_quoted_image_uses_its_id_and_a_missing_quote_adds_none(self):
        self.start(unnamed=False)
        self._look()
        red, blue = self._png("red.png", "red"), self._png("blue.png", "blue")
        self._picture(red, identifier="83", sequence=1)
        self._picture(blue, sender=OTHER, identifier="81", sequence=2)
        quiet = Message("看看", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=9, type=49,
                        data={"app_type": 57, "quoted_message_id": "81",
                              "quoted_message": {"author": "别人", "kind": "image", "text": "[图片]",
                                                 "available": True, "truncated": False}})
        self.store.record(quiet)
        self.plugin.on_message(quiet, self.ctx)
        self.assertEqual(self.offered, [])
        asked = Message("看看", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=10, type=49,
                        data={"app_type": 57, "is_at_self": True, "at_members": (ROBOT,), "quoted_message_id": "81",
                              "quoted_message": {"author": "别人", "kind": "image", "text": "[图片]",
                                                 "available": True, "truncated": False}})
        self.store.record(asked)
        self.plugin.on_message(asked, self.ctx)
        self.assertIn(vision_data_url(blue), self.looked)
        missing = Message("再看", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=11, type=49,
                          data={"app_type": 57, "is_at_self": True, "at_members": (ROBOT,), "quoted_message_id": "404",
                                "quoted_message": {"author": "别人", "kind": "image", "text": "[图片]",
                                                   "available": True, "truncated": False}})
        self.store.record(missing)
        self.plugin.on_message(missing, self.ctx)
        ids = [row.get("id") for row in self.payloads[-1]["transcript"] if row.get("type") == "image"]
        self.assertNotIn("404", ids)

    def test_quoted_image_waits_until_the_download_is_ready(self):
        from wxbot_framework.image_receiver import _track_download
        self.start(unnamed=False)
        self._look()
        red = self._png("late.png", "red")
        _track_download("91", True)

        def finish():
            time.sleep(0.15)
            self._picture(red, identifier="91", sequence=1)
            _track_download("91", False)

        threading.Thread(target=finish, daemon=True).start()
        asked = Message("看看", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=2, type=49,
                        data={"app_type": 57, "is_at_self": True, "at_members": (ROBOT,), "quoted_message_id": "91",
                              "quoted_message": {"author": "小猫", "kind": "image", "text": "[图片]",
                                                 "available": True, "truncated": False}})
        self.store.record(asked)
        self.plugin.on_message(asked, self.ctx)
        deadline = time.perf_counter() + 2
        while time.perf_counter() < deadline and not self.looked:
            time.sleep(0.05)
        self.assertIn(vision_data_url(red), self.looked)
