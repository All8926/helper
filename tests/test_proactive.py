"""主动插话与接话识别：本地先筛选，活跃度大于 0 的群才会再问模型。"""
import unittest
from datetime import timedelta
from unittest.mock import Mock

import test_chat_summary as daily
from plugins.mention_chat import Plugin
from wxbot_framework import proactive as pro
from wxbot_framework.room_behavior import BehaviorError, normalize_room_setting
from wxbot_framework.sdk import Message

ROOM, CAT, ROBOT, OTHER = daily.ROOM, daily.CAT, daily.ROBOT, daily.OTHER


class FilterTests(unittest.TestCase):
    def test_activity_decides_which_moments_count(self):
        quiet = [{"sender_id": CAT, "text": "哈哈", "bot": False, "id": 1}]
        self.assertEqual(pro.candidate_reason("哈哈", quiet, bot_id=ROBOT, level=10), "")
        self.assertEqual(pro.candidate_reason("这个怎么弄？", quiet, bot_id=ROBOT, level=2), "")
        self.assertEqual(pro.candidate_reason("这个怎么弄？", quiet, bot_id=ROBOT, level=3), "question")
        fight = [{"sender_id": CAT, "text": "你闭嘴", "bot": False, "id": 1},
                 {"sender_id": OTHER, "text": "你才废物", "bot": False, "id": 2}]
        self.assertEqual(pro.candidate_reason("行吧", fight, bot_id=ROBOT, level=3), "mediate")
        waiting = [{"sender_id": CAT, "text": "有人知道吗", "bot": False, "id": 4, "age_seconds": 20}]
        self.assertEqual(pro.candidate_reason("我到了", waiting, current_id=9, bot_id=ROBOT, level=1), "unanswered")
        self.assertEqual(pro.base_cooldown(1), 1800)
        self.assertEqual(pro.base_cooldown(10), 45)
        self.assertGreater(pro.base_cooldown(4), pro.base_cooldown(6))

    def test_activity_setting_defaults_off_and_old_switch_maps_to_five(self):
        self.assertEqual(normalize_room_setting({}, [])["activity"], 0)
        self.assertEqual(normalize_room_setting({"proactive_enabled": True}, [])["activity"], 5)
        self.assertEqual(normalize_room_setting({"activity": 8, "proactive_enabled": True}, [])["activity"], 8)
        with self.assertRaises(BehaviorError):
            normalize_room_setting({"activity": 11}, [])

    def test_ignored_lines_widen_the_gap_and_complaints_mute(self):
        gate = pro.ProactiveGate()
        gate.mark(ROOM, now=0)
        self.assertTrue(gate.allow(ROOM, 60, 0, now=190))
        self.assertEqual(gate.scale(ROOM), 2)
        gate.engaged(ROOM)
        self.assertEqual(gate.scale(ROOM), 1)
        gate.mute(ROOM, now=200)
        self.assertFalse(gate.allow(ROOM, 60, 0, now=300))
        self.assertTrue(gate.allow(ROOM, 60, 0, now=3801))
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
        self.assertEqual(len(pro.clean_reply("啊" * 100)), 80)


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

    def start(self, activity=5):
        self.store.room_settings[ROOM] = {"activity": activity}
        self.backend.capabilities = ("text.quote.send",)
        self.backend.send_quote_text = lambda room, original_id, text: self.backend.sent.append((room, text, original_id)) or True
        self.backend.send_at_text = lambda room, text, ids: self.backend.sent.append((room, text, ids)) or True
        self.interject = {"reply": "我帮你看一眼", "quote": False}
        self.follow = {"to_bot": True, "reply": "明天也行"}
        self.judged, self.followed = [], []

        def provider(system, user):
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

    def test_model_can_quote_the_person_it_answers(self):
        self.start()
        self.interject = {"reply": "试试先重启", "quote": True}
        self.say("这个怎么弄？", 1)
        self.assertEqual(self.backend.sent, [(ROOM, "试试先重启", 1)])

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

    def test_complaint_mutes_proactive_lines(self):
        self.start()
        self.say("这个怎么弄？", 1)
        self.say("闭嘴吧", 2, sender=OTHER)
        self.now += timedelta(minutes=10)
        self.say("还有谁会修电脑？", 3, sender=OTHER)
        self.assertEqual(len(self.judged), 1)

    def test_quiet_group_gets_one_follow_up_per_silence(self):
        self.start()
        self.store.record(self.message("我们今晚吃火锅", sequence=1))
        self.now += timedelta(minutes=1)
        self.plugin._cold_once()
        self.assertEqual(self.judged, [])
        self.now += timedelta(minutes=4)
        self.interject = {"reply": "火锅也不错", "quote": True}
        self.plugin._cold_once()
        self.assertEqual(self.backend.sent, [(ROOM, "火锅也不错")])
        self.assertIn("安静", self.judged[0][0])
        self.plugin._cold_once()
        self.assertEqual(len(self.judged), 1)

    def test_low_activity_skips_quiet_group(self):
        self.start(activity=3)
        self.store.record(self.message("我们今晚吃火锅", sequence=1))
        self.now += timedelta(minutes=5)
        self.plugin._cold_once()
        self.assertEqual(self.judged, [])
