"""主动插话与接话识别：本地先算开口意愿分，过了本群门槛才问模型；群里在聊时定时让模型看一眼。"""
import time
import unittest
from datetime import timedelta
from unittest.mock import Mock

from PIL import Image

import test_chat_summary as daily
from plugins.mention_chat import Plugin
from wxbot_framework import proactive as pro
from wxbot_framework.llm import LLMError
from wxbot_framework.room_behavior import BehaviorError, normalize_reply_limit, normalize_room_setting
from wxbot_framework.sdk import Message
from wxbot_framework.vision import vision_data_url

ROOM, CAT, ROBOT, OTHER = daily.ROOM, daily.CAT, daily.ROBOT, daily.OTHER


class FilterTests(unittest.TestCase):
    def test_score_follows_need_and_drops_when_bot_just_spoke(self):
        quiet = [{"sender_id": CAT, "text": "哈哈", "bot": False, "id": 1}]
        self.assertEqual(pro.willingness("哈哈", quiet, bot_id=ROBOT), (0.0, ""))
        asked, reason = pro.willingness("这个怎么弄？", quiet, bot_id=ROBOT)
        self.assertEqual(reason, "question")
        helped, _ = pro.willingness("求助，这个怎么弄？", quiet, bot_id=ROBOT)
        self.assertGreater(helped, asked)
        fight = [{"sender_id": CAT, "text": "你闭嘴", "bot": False, "id": 1},
                 {"sender_id": OTHER, "text": "你才废物", "bot": False, "id": 2}]
        self.assertEqual(pro.willingness("行吧", fight, bot_id=ROBOT)[1], "mediate")
        short = [{"sender_id": CAT, "text": "有人知道吗", "bot": False, "id": 4, "age_seconds": 20}]
        long = [dict(short[0], age_seconds=120)]
        self.assertGreater(pro.willingness("我到了", long, current_id=9, bot_id=ROBOT)[0],
                           pro.willingness("我到了", short, current_id=9, bot_id=ROBOT)[0])
        after_bot = quiet + [{"sender_id": ROBOT, "text": "我在", "bot": True, "id": 2}]
        self.assertLess(pro.willingness("这个怎么弄？", after_bot, bot_id=ROBOT)[0], asked - 0.3)

    def test_activity_sets_share_and_bar(self):
        self.assertAlmostEqual(pro.target_share(1), 0.01)
        self.assertAlmostEqual(pro.target_share(10), 0.15)
        self.assertAlmostEqual(pro.speak_bar(1, 1.0), 0.70)
        self.assertAlmostEqual(pro.speak_bar(10, 2.0), 0.25)
        self.assertAlmostEqual(pro.speak_bar(5, 0.0), 0.90)
        self.assertAlmostEqual(pro.speak_bar(5, 1.0, 0.2), 0.70)
        self.assertEqual(pro.glance_interval(2), 120)
        self.assertEqual(pro.glance_interval(2, 0.4), 360)

    def test_settings_default_and_validate(self):
        self.assertEqual(normalize_room_setting({}, [])["activity"], 0)
        self.assertEqual(normalize_room_setting({"proactive_enabled": True}, [])["activity"], 5)
        self.assertEqual(normalize_room_setting({}, [])["glance_minutes"], 2)
        self.assertTrue(normalize_room_setting({}, [])["deep_thinking"])
        self.assertFalse(normalize_room_setting({"deep_thinking": False}, [])["deep_thinking"])
        self.assertEqual(normalize_room_setting({"glance_minutes": 0}, [])["glance_minutes"], 0)
        self.assertEqual(normalize_reply_limit(None), 600)
        self.assertEqual(normalize_reply_limit(8000), 8000)
        for bad in (0, 8001, True, 1.5, "600"):
            with self.assertRaises(BehaviorError):
                normalize_reply_limit(bad)
        for bad in ({"activity": 11}, {"glance_minutes": 61}, {"deep_thinking": "yes"}):
            with self.assertRaises(BehaviorError):
                normalize_room_setting(bad, [])

    def test_budget_feedback_and_mute(self):
        gate = pro.ProactiveGate()
        gate.earn(ROOM, 5)
        self.assertEqual(gate.tokens(ROOM), pro.BUDGET_CAP)
        gate.mark(ROOM, now=0)
        self.assertEqual(gate.tokens(ROOM), pro.BUDGET_CAP - 1)
        self.assertFalse(gate.allow(ROOM, 0, now=30))
        self.assertTrue(gate.allow(ROOM, 0, now=61))
        self.assertFalse(gate.settle(ROOM, now=100))
        self.assertTrue(gate.settle(ROOM, now=181))
        self.assertFalse(gate.settle(ROOM, now=400))
        gate.mark(ROOM, now=500)
        self.assertTrue(gate.engaged(ROOM))
        self.assertFalse(gate.engaged(ROOM))
        gate.mute(ROOM, now=600)
        self.assertFalse(gate.allow(ROOM, 0, now=700))
        self.assertTrue(gate.allow(ROOM, 0, now=4201))
        recent = [{"sender_id": ROBOT, "text": "我来说两句", "bot": True, "id": 1}]
        self.assertTrue(pro.annoyed("闭嘴吧你", recent, 2, ROBOT))
        self.assertFalse(pro.annoyed("闭嘴吧你", [{"sender_id": OTHER, "text": "哈", "id": 1}], 2, ROBOT))

    def test_follow_up_window_and_turn_limit(self):
        sessions = pro.FollowUps()
        sessions.open(ROOM, CAT, now=0)
        self.assertTrue(sessions.active(ROOM, CAT, now=100))
        self.assertFalse(sessions.active(ROOM, CAT, now=200))
        for moment in range(5):
            sessions.advance(ROOM, CAT, now=moment)
        self.assertFalse(sessions.active(ROOM, CAT, now=10))
        self.assertTrue(sessions.exhausted(ROOM, CAT, now=10))
        recent = [{"sender_id": ROBOT, "text": "火锅", "bot": True, "id": 1, "age_seconds": 30}]
        self.assertTrue(pro.addresses_bot("你推荐哪家", recent, 2, ROBOT))
        self.assertFalse(pro.addresses_bot("我推荐哪家", recent, 2, ROBOT))
        self.assertEqual(pro.clean_reply("不说。"), "")
        self.assertEqual(len(pro.clean_reply("啊" * 100)), 100)

    def test_glance_needs_fresh_lines_while_chat_is_live(self):
        lines = [{"sender_id": CAT, "text": "去爬山", "id": 1, "age_seconds": 40},
                 {"sender_id": OTHER, "text": "好啊", "id": 2, "age_seconds": 30}]
        self.assertTrue(pro.glance_due(lines, ROBOT, None))
        self.assertFalse(pro.glance_due(lines, ROBOT, 2))
        self.assertFalse(pro.glance_due([dict(row, age_seconds=200) for row in lines], ROBOT, None))
        self.assertFalse(pro.glance_due(lines + [{"sender_id": ROBOT, "text": "我也去", "bot": True, "id": 3,
                                                  "age_seconds": 10}], ROBOT, None))


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

    def start(self, activity=5, **setting):
        self.store.room_settings[ROOM] = {"activity": activity, **setting}
        self.backend.capabilities = ("text.quote.send",)
        self.backend.send_quote_text = lambda room, original_id, text: self.backend.sent.append((room, text, original_id)) or True
        self.backend.send_at_text = lambda room, text, ids: self.backend.sent.append((room, text, ids)) or True
        self.interject = {"reply": "我帮你看一眼", "quote": False}
        self.follow = {"to_bot": True, "reply": "明天也行"}
        self.judged, self.followed = [], []

        def provider(system, user, **_kwargs):
            if "to_bot" in system:
                self.followed.append(user)
                return self.follow
            if "没人点你的名" in system:
                self.judged.append((system, user))
                return self.interject() if callable(self.interject) else self.interject
            return {"reply": "这是回答。"}

        self.ctx._llm = Mock(complete_json=provider)
        self.plugin = Plugin()
        self.plugin.on_load(self.ctx)

    def say(self, text, sequence, sender=CAT, **data):
        msg = self.message(text, sender=sender, sequence=sequence, data={"native_message_id": sequence, **data})
        self.store.record(msg)
        self.plugin.on_message(msg, self.ctx)
        return msg

    def at(self, text, sequence, sender=CAT):
        return self.say("@机器人\u2005" + text, sequence, sender=sender, at_members=(ROBOT,), is_at_self=True)

    def test_zero_activity_and_chatter_never_call_the_model(self):
        self.start(activity=0)
        self.say("这个怎么弄？", 1)
        self.store.room_settings[ROOM] = {"activity": 5}
        self.say("哈哈", 2)
        self.assertEqual(self.judged, [])
        self.assertEqual(self.backend.sent, [])

    def test_question_gets_one_line_with_full_context_then_bot_waits(self):
        self.start()
        self.store.replace_facts(ROOM, CAT, ("喜欢火锅",), 0)
        self.say("这个怎么弄？", 1)
        system, user = self.judged[0]
        self.assertIn("禁止聊政治", system)
        self.assertIn("这个怎么弄？", user)
        self.assertIn("your_recent", user)
        self.assertEqual(self.backend.sent, [(ROOM, "我帮你看一眼")])
        self.say("那下一步呢", 2, sender=OTHER)
        self.assertEqual(len(self.judged), 1)

    def test_low_activity_holds_back_a_plain_question(self):
        self.start(activity=2)
        self.say("这个怎么弄？", 1)
        self.assertEqual(self.judged, [])

    def test_proactive_line_stays_plain_even_if_model_asks_to_quote(self):
        self.start()
        self.interject = {"reply": "试试先重启", "quote": True}
        self.say("这个怎么弄？", 1)
        self.assertEqual(self.backend.sent, [(ROOM, "试试先重启")])

    def test_someone_answering_during_the_call_cancels_the_line(self):
        self.start()

        def answered():
            self.store.record(self.message("先拧松螺丝", sender=OTHER, sequence=7))
            return {"reply": "我帮你看一眼", "quote": False}

        self.interject = answered
        self.say("这个怎么弄？", 1)
        self.assertEqual(len(self.judged), 1)
        self.assertEqual(self.backend.sent, [])

    def test_follow_up_without_mention_is_recognised_and_rejected_when_not_for_bot(self):
        self.start()
        self.at("今晚吃什么", 1)
        self.assertEqual(self.backend.sent[-1], (ROOM, "这是回答。", 1))
        self.say("那明天吃啥", 2)
        self.assertEqual(len(self.followed), 1)
        self.assertEqual(self.backend.sent[-1], (ROOM, "明天也行", 2))
        self.follow = {"to_bot": False, "reply": ""}
        self.say("我跟你们说个事", 3)
        self.assertEqual(len(self.followed), 2)
        self.assertEqual(len(self.backend.sent), 2)
        self.say("今天降温了", 4)
        self.assertEqual(len(self.followed), 2)

    def test_follow_up_stays_off_when_activity_is_zero(self):
        self.start(activity=0)
        self.at("今晚吃什么", 1)
        self.say("那明天吃啥", 2)
        self.assertEqual(self.followed, [])
        self.assertEqual(len(self.backend.sent), 1)

    def test_quoting_a_bot_line_gets_a_reply(self):
        self.start(activity=0)
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
        self.start(activity=0)
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

    def test_complaint_mutes_and_raises_the_bar(self):
        self.start()
        self.say("这个怎么弄？", 1)
        self.say("闭嘴吧", 2, sender=OTHER)
        self.assertAlmostEqual(self.store.proactive_bias(ROOM), pro.ANNOYED_STEP)
        self.now += timedelta(minutes=10)
        self.say("还有谁会修电脑？", 3, sender=OTHER)
        self.assertEqual(len(self.judged), 1)

    def test_reactions_move_the_room_bar_and_persist(self):
        self.start()
        self.say("这个怎么弄？", 1)
        self.at("谢谢你", 2, sender=OTHER)
        self.assertAlmostEqual(self.store.proactive_bias(ROOM), pro.ENGAGED_STEP)
        self.plugin.gate.mark(ROOM)
        self.plugin._settle(self.ctx, ROOM, now=time.monotonic() + pro.IGNORE_WINDOW + 1)
        self.assertAlmostEqual(self.store.proactive_bias(ROOM), pro.ENGAGED_STEP + pro.IGNORED_STEP)
        self.assertEqual(self.store.shift_proactive_bias(ROOM, 5, pro.BIAS_LOW, pro.BIAS_HIGH), pro.BIAS_HIGH)

    def test_quiet_group_gets_one_follow_up_per_silence(self):
        self.start(activity=10)
        self.store.record(self.message("我们今晚吃火锅", sequence=1))
        self.now += timedelta(minutes=1)
        self.plugin._watch_once()
        self.assertEqual(self.judged, [])
        self.now += timedelta(minutes=4)
        self.interject = {"reply": "火锅也不错", "quote": True}
        self.plugin._watch_once()
        self.assertEqual(self.backend.sent, [(ROOM, "火锅也不错")])
        self.assertIn("安静", self.judged[0][0])
        self.plugin._watch_once()
        self.assertEqual(len(self.judged), 1)

    def test_low_activity_skips_quiet_group(self):
        self.start(activity=3)
        self.store.record(self.message("我们今晚吃火锅", sequence=1))
        self.now += timedelta(minutes=5)
        self.plugin._watch_once()
        self.assertEqual(self.judged, [])

    def test_glance_runs_on_its_interval_and_spends_budget(self):
        self.start(activity=5, glance_minutes=2)
        self.interject = {"reply": "我推荐香山", "quote": False}
        self.store.record(self.message("我们周末去爬山吧", sequence=1))
        self.store.record(self.message("好啊去哪座山", sender=OTHER, sequence=2))
        self.plugin._watch_once(now=1000)
        self.assertIn("隔了一会儿", self.judged[0][0])
        self.assertEqual(self.backend.sent, [(ROOM, "我推荐香山")])
        for sequence, text in ((3, "香山人多吗"), (4, "周末肯定多"), (5, "那早点出发")):
            self.store.record(self.message(text, sequence=sequence))
        self.plugin._watch_once(now=1010)
        self.plugin._watch_once(now=1121)
        self.assertEqual(len(self.judged), 1)
        self.plugin.gate.earn(ROOM, 1)
        self.plugin._watch_once(now=1130)
        self.assertEqual(len(self.judged), 2)

    def test_glance_off_when_interval_is_zero(self):
        self.start(activity=10, glance_minutes=0)
        self.store.record(self.message("我们周末去爬山吧", sequence=1))
        self.store.record(self.message("好啊去哪座山", sender=OTHER, sequence=2))
        self.plugin._watch_once(now=1000)
        self.assertEqual(self.judged, [])

    def _png(self, name, color):
        path = self.root / name
        Image.new("RGB", (4, 4), color).save(path)
        return path

    def _picture(self, path, sender=CAT, identifier="88", status="ready"):
        self.plugin.on_message(Message("", ROOM, sender_id=sender, is_group=True, session_id="test",
            sequence=1, type=3, data={"image_status": status, "image_path": str(path),
                                      "native_message_id": identifier}), self.ctx)

    def _capture(self, provider):
        self.ctx._llm.complete_json = provider

    def test_mention_sends_only_that_persons_recent_image(self):
        self.start()
        seen = []

        def provider(system, user, image=None, **_kwargs):
            seen.append(image)
            if "to_bot" in system:
                return {"to_bot": True, "reply": "接着说。"}
            return {"reply": "看到了。"}

        self._capture(provider)
        red, blue = self._png("red.png", "red"), self._png("blue.png", "blue")
        self._picture(blue, sender=OTHER, identifier="81")
        self.at("看别人的图", 2)
        self.assertIsNone(seen[-1])
        self._picture(red, status="failed", identifier="82")
        self.at("失败的图", 4)
        self.assertIsNone(seen[-1])
        self._picture(red, identifier="83")
        self.at("这是什么", 6)
        self.assertEqual(seen[-1], vision_data_url(red))
        self.say("那左边是什么意思", 7)
        self.assertIsNone(seen[-1])

    def test_quoted_image_wins_and_a_missing_quote_does_not_substitute(self):
        self.start(activity=0)
        seen = []

        def provider(system, user, image=None, **_kwargs):
            seen.append((image, "附带一张图片" in system, '"image": true' in user))
            return {"reply": "看到了。"}

        self._capture(provider)
        red, blue = self._png("red.png", "red"), self._png("blue.png", "blue")
        self._picture(red, identifier="83")
        self._picture(blue, sender=OTHER, identifier="81")
        quiet = Message("看看", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=9, type=49,
                        data={"app_type": 57, "quoted_message_id": "81",
                              "quoted_message": {"author": "别人", "kind": "image", "text": "[图片]",
                                                 "available": True, "truncated": False}})
        self.store.record(quiet)
        self.plugin.on_message(quiet, self.ctx)
        self.assertEqual(seen, [])
        asked = Message("看看", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=10, type=49,
                        data={"app_type": 57, "is_at_self": True, "at_members": (ROBOT,), "quoted_message_id": "81",
                              "quoted_message": {"author": "别人", "kind": "image", "text": "[图片]",
                                                 "available": True, "truncated": False}})
        self.store.record(asked)
        self.plugin.on_message(asked, self.ctx)
        self.assertEqual(seen[-1], (vision_data_url(blue), True, True))
        missing = Message("再看", ROOM, sender_id=CAT, is_group=True, session_id="test", sequence=11, type=49,
                          data={"app_type": 57, "is_at_self": True, "at_members": (ROBOT,), "quoted_message_id": "404",
                                "quoted_message": {"author": "别人", "kind": "image", "text": "[图片]",
                                                   "available": True, "truncated": False}})
        self.store.record(missing)
        self.plugin.on_message(missing, self.ctx)
        self.assertEqual(seen[-1], (None, False, False))

    def test_rejected_image_falls_back_to_text_and_other_errors_do_not(self):
        self.start(activity=0)
        path = self._png("red.png", "red")
        calls = []

        def reject(system, user, image=None, **_kwargs):
            calls.append(image)
            if image:
                raise LLMError("大模型接口 HTTP 400；请核对密钥、余额和模型配置", status=400)
            return {"reply": "只看文字。"}

        self._capture(reject)
        self._picture(path, identifier="83")
        self.at("这是什么", 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(str(calls[0]).startswith("data:image/png;base64,"))
        self.assertIsNone(calls[1])
        self.assertIn("只看文字。", self.backend.sent[-1][1])

        calls.clear()
        self.backend.sent.clear()

        def denied(system, user, image=None):
            calls.append(image)
            raise LLMError("大模型接口 HTTP 401；请核对密钥、余额和模型配置", status=401)

        self._capture(denied)
        self.at("再看一次", 3)
        self.assertEqual(len(calls), 1)
        self.assertIn("暂时无法回复", self.backend.sent[-1][1])
