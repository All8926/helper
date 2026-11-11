"""主动插话：本地先筛选，打开的群才会再问一次模型。"""
import unittest
from datetime import timedelta
from unittest.mock import Mock

import test_chat_summary as daily
from plugins.mention_chat import Plugin
from wxbot_framework.proactive import ProactiveGate, candidate_reason, clean_reply
from wxbot_framework.room_behavior import BehaviorError, normalize_room_setting

ROOM, CAT, ROBOT, OTHER = daily.ROOM, daily.CAT, daily.ROBOT, daily.OTHER


class FilterTests(unittest.TestCase):
    def test_only_questions_conflict_and_open_topics_are_candidates(self):
        quiet = [{"sender_id": CAT, "text": "哈哈", "bot": False, "id": 1}]
        self.assertEqual(candidate_reason("哈哈", quiet, bot_id=ROBOT), "")
        self.assertEqual(candidate_reason("这个怎么弄？", quiet, bot_id=ROBOT), "question")
        fight = [
            {"sender_id": CAT, "text": "你闭嘴", "bot": False, "id": 1},
            {"sender_id": OTHER, "text": "你才废物", "bot": False, "id": 2},
        ]
        self.assertEqual(candidate_reason("行吧", fight, bot_id=ROBOT), "mediate")
        thread = [{"sender_id": ROBOT, "text": "我刚说了火锅", "bot": True, "id": 3}]
        self.assertEqual(candidate_reason("接着说", thread, current_id=9, bot_id=ROBOT), "bot_thread")
        waiting = [{"sender_id": CAT, "text": "有人知道吗", "bot": False, "id": 4, "age_seconds": 20}]
        self.assertEqual(candidate_reason("我到了", waiting, current_id=9, bot_id=ROBOT), "unanswered")

    def test_recent_bot_line_blocks_and_reply_noise_is_dropped(self):
        recent = [
            {"sender_id": OTHER, "text": "今晚吃什么", "bot": False},
            {"sender_id": ROBOT, "text": "火锅", "bot": True},
        ]
        from wxbot_framework.proactive import bot_spoke_recently
        self.assertTrue(bot_spoke_recently(recent, ROBOT))
        self.assertEqual(clean_reply("  不说  "), "")
        self.assertEqual(clean_reply("no"), "")
        self.assertEqual(len(clean_reply("啊" * 50)), 40)

    def test_missing_switch_stays_off_and_cooldown_is_bounded(self):
        setting = normalize_room_setting({}, [])
        self.assertFalse(setting["proactive_enabled"])
        self.assertEqual(setting["proactive_cooldown"], 180)
        gate = ProactiveGate()
        self.assertTrue(gate.allow(ROOM, 180, 30, now=100))
        self.assertFalse(gate.allow(ROOM, 180, 30, now=120))
        gate.mark(ROOM, now=100)
        self.assertFalse(gate.allow(ROOM, 180, 30, now=200))
        self.assertTrue(gate.allow(ROOM, 180, 30, now=281))
        with self.assertRaises(BehaviorError):
            normalize_room_setting({"proactive_cooldown": 10}, [])


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

    def enable(self):
        self.store.room_settings[ROOM] = {"proactive_enabled": True, "proactive_cooldown": 180}
        provider = Mock(return_value={"reply": "我帮你看一眼"})
        self.ctx._llm = Mock(complete_json=provider)
        self.plugin = Plugin()
        self.plugin.on_load(self.ctx)
        return provider

    def test_closed_group_and_chatter_do_not_call_the_model(self):
        provider = self.enable()
        self.store.room_settings[ROOM] = {"proactive_enabled": False}
        msg = self.message("这个怎么弄？", sequence=1)
        self.store.record(msg)
        self.plugin.on_message(msg, self.ctx)
        self.store.room_settings[ROOM]["proactive_enabled"] = True
        chat = self.message("哈哈", sequence=2)
        self.store.record(chat)
        self.plugin.on_message(chat, self.ctx)
        provider.assert_not_called()
        self.assertEqual(self.backend.sent, [])

    def test_question_sends_one_plain_line_and_bot_recent_stays_quiet(self):
        provider = self.enable()
        msg = self.message("这个怎么弄？", sequence=1)
        self.store.record(msg)
        self.plugin.on_message(msg, self.ctx)
        system, user = provider.call_args.args
        self.assertIn("禁止聊政治", system)
        self.assertIn("这个怎么弄？", user)
        self.assertEqual(self.backend.sent, [(ROOM, "我帮你看一眼")])
        follow = self.message("那下一步呢", sequence=2)
        self.store.record(follow)
        self.plugin.on_message(follow, self.ctx)
        self.assertEqual(provider.call_count, 1)

    def test_empty_reply_does_not_send(self):
        provider = self.enable()
        provider.return_value = {"reply": ""}
        msg = self.message("这个怎么弄？", sequence=1)
        self.store.record(msg)
        self.plugin.on_message(msg, self.ctx)
        self.assertEqual(self.backend.sent, [])
        self.assertEqual(provider.call_count, 1)

    def test_cold_room_sends_one_plain_follow_up(self):
        provider = self.enable()
        msg = self.message("我们今晚吃火锅", sequence=1)
        self.store.record(msg)
        self.now += timedelta(minutes=5)
        provider.return_value = {"reply": "火锅也不错"}
        self.plugin._cold_once()
        self.assertEqual(self.backend.sent, [(ROOM, "火锅也不错")])
        self.assertIn("安静", provider.call_args.args[0])
        self.plugin._cold_once()
        self.assertEqual(provider.call_count, 1)

    def test_cold_path_waits_until_the_group_has_been_quiet(self):
        provider = self.enable()
        msg = self.message("我们今晚吃火锅", sequence=1)
        self.store.record(msg)
        self.now += timedelta(minutes=1)
        self.plugin._cold_once()
        self.now += timedelta(minutes=40)
        self.plugin._cold_once()
        provider.assert_not_called()
        self.assertEqual(self.backend.sent, [])
