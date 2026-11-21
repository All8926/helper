"""私聊名单、流水和回复入口。不连接微信。"""
import json
import logging
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

from fake_llm import FakeLLM
from plugins.private_chat import Plugin
from test_web_panel import PanelFixture
from wxbot_framework.llm import LLMError
from wxbot_framework.members import MemberError, MemberStore
from wxbot_framework.personality import PersonalityLLM
from wxbot_framework.room_behavior import BehaviorError
from wxbot_framework.sdk import Message

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / ".runtime/tests"
ROOM = "12345678@chatroom"
ROBOT, FRIEND, OTHER = "wxid_robot", "wxid_friend", "wxid_other"


class PrivateFixture:
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / "members.json"
        self.path.write_text(json.dumps({
            "database_file": "data.sqlite3", "history_enabled": True, "daily_clear": True,
            "groups": {ROOM: []},
        }, ensure_ascii=False), encoding="utf-8")
        self.now = datetime(2026, 11, 15, 1, tzinfo=timezone.utc)
        self.store = MemberStore(self.path, ROBOT, now=lambda: self.now)
        self.store.apply_private_chat({
            "contacts": [FRIEND], "names": {FRIEND: "朋友"},
            "context_hours": 12, "context_limit": 40, "personality": "默认"})

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def message(self, text="你好", sender=FRIEND, sequence=1, **kwargs):
        return Message(text, FRIEND, sender_id=sender, session_id="one", sequence=sequence, **kwargs)

class PrivateStoreTests(PrivateFixture, unittest.TestCase):
    def test_private_history_is_kept_out_of_group_watch_and_group_diary(self):
        self.assertNotIn(FRIEND, self.store.groups)
        self.assertTrue(self.store.record(self.message()))
        self.assertFalse(self.store.record(Message("旁人", OTHER, sender_id=OTHER, session_id="one")))
        self.assertFalse(self.store.record(Message("另一群", "99999@chatroom", sender_id=FRIEND, is_group=True, session_id="one")))
        self.assertTrue(self.store.record(Message("群里", ROOM, sender_id=FRIEND, is_group=True, session_id="one", sequence=2)))
        self.assertEqual(self.store.recent_room_lines(FRIEND, 3600, 10), [])
        found = self.store.search_room_lines(FRIEND, "你好", 1)
        self.assertEqual([(row["speaker"], row["text"]) for row in found], [("朋友", "你好")])
        self.assertTrue(self.store.record_bot_reply(self.message(), "收到，这是私聊回答"))
        reply = self.store.search_room_lines(FRIEND, "私聊回答", 1)
        self.assertEqual(reply[-1]["speaker"], "机器人")
        self.now += timedelta(days=1)
        self.store.rollover()
        rooms = [room for room, _, _ in self.store.pending_digests()]
        self.assertIn(ROOM, rooms)
        self.assertNotIn(FRIEND, rooms)

    def test_shared_personality_changes_generation_and_rejects_non_contacts(self):
        self.assertEqual(self.store.private_context_window(), (12, 40))
        self.store.apply_personalities([
            {"name": name, "layers": {"声口": "短。"}, "previous": name}
            for name in ("嘴臭", "卑微", "夸夸")
        ])
        first = self.store.private_personality(FRIEND)
        self.store.apply_private_chat({
            "contacts": [FRIEND], "context_hours": 12, "context_limit": 40, "personality": "嘴臭"})
        chosen = self.store.private_personality(FRIEND)
        self.assertEqual(chosen.name, "嘴臭")
        self.assertGreater(chosen.generation, first.generation)
        again = self.store.apply_private_chat({
            "contacts": [FRIEND], "context_hours": 12, "context_limit": 40, "personality": "嘴臭"})
        self.assertEqual(self.store.private_personality(FRIEND).generation, chosen.generation)
        self.assertEqual(again["names"], {})
        self.store.apply_personalities([
            {"name": "卑微", "layers": {"声口": "让。"}, "previous": "卑微"},
            {"name": "夸夸", "layers": {"声口": "夸。"}, "previous": "夸夸"},
        ])
        reset = self.store.private_personality(FRIEND)
        self.assertEqual(reset.name, "默认")
        self.assertGreater(reset.generation, chosen.generation)
        self.store.apply_private_chat({
            "contacts": [FRIEND, OTHER], "context_hours": 12, "context_limit": 40,
            "people": {
                FRIEND: {"name": "甲", "personality": "卑微", "deep_thinking": False,
                         "voice_mode": "auto", "voice_threshold": 0.2, "sing_threshold": 0.4},
                OTHER: {"name": "", "personality": "夸夸"}}})
        friend = self.store.private_personality(FRIEND)
        other = self.store.private_personality(OTHER)
        self.assertEqual(friend.name, "卑微")
        self.assertEqual(other.name, "夸夸")
        self.assertNotEqual(friend.room, other.room)
        self.assertFalse(self.store.room_deep_thinking(FRIEND))
        self.assertTrue(self.store.room_deep_thinking(OTHER))
        self.assertEqual(self.store.room_voice_threshold(FRIEND), 0.2)
        self.assertEqual(self.store.room_sing_threshold(FRIEND), 0.4)
        self.assertTrue(self.store.voice_selection(FRIEND).enabled)
        self.assertEqual(self.store.voice_selection(FRIEND).mode, "auto")
        self.assertFalse(self.store.voice_selection(OTHER).enabled)
        self.store.apply_private_chat({
            "contacts": [FRIEND, OTHER], "context_hours": 12, "context_limit": 40,
            "people": {
                FRIEND: {"name": "甲", "personality": "夸夸", "deep_thinking": False, "voice_mode": "auto"},
                OTHER: {"personality": "夸夸"}}})
        self.assertGreater(self.store.private_personality(FRIEND).generation, friend.generation)
        self.assertEqual(self.store.private_personality(OTHER).generation, other.generation)
        with self.assertRaises(MemberError):
            self.store.apply_private_chat({"contacts": [ROBOT]})
        with self.assertRaises(BehaviorError):
            self.store.apply_private_chat({"contacts": ["filehelper"]})
        with self.assertRaises(BehaviorError):
            self.store.apply_private_chat({"contacts": [ROOM]})

    def test_snapshot_uses_configured_name_then_wechat_nickname(self):
        self.store.record(self.message())
        named = self.store.conversation_snapshot(self.message())
        self.assertEqual(named.speaker_name, "朋友")
        self.assertEqual(named.current, "你好")
        self.store.apply_private_chat({
            "contacts": [FRIEND], "names": {FRIEND: ""}, "context_hours": 12, "context_limit": 40})
        self.store.remember_wechat_name(FRIEND, "好友甲")
        self.store.record(self.message("再看一次", sequence=2))
        fallback = self.store.conversation_snapshot(self.message("再看一次", sequence=2))
        self.assertEqual(fallback.speaker_name, "好友甲")
        self.assertEqual([row["text"] for row in fallback.messages], ["你好"])


class PrivatePluginTests(PrivateFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.llm = FakeLLM(chat=Mock(return_value="收到，这是私聊回答"))
        self.ctx = Mock()
        self.ctx.members = self.store
        self.ctx.logger = logging.getLogger("private-test")
        self.ctx.llm = self.llm
        self.ctx.llm_for = lambda message, personality=None: PersonalityLLM(
            self.llm, personality, current=lambda: self.store.personality_is_current(personality),
            prompts=self.store.prompts(), thinking=True)
        self.ctx.reply = Mock(return_value=True)
        self.ctx.response_guard = self.guard
        self.plugin = Plugin()
        self.plugin.on_load(self.ctx)

    @staticmethod
    @contextmanager
    def guard(snapshot, personality):
        yield

    def test_only_listed_private_text_is_answered_with_private_tools(self):
        ignored = (
            Message("群里说话", ROOM, sender_id=FRIEND, is_group=True, session_id="one", sequence=1),
            Message("你好", OTHER, sender_id=OTHER, session_id="one", sequence=1),
            Message("/生图 猫", FRIEND, sender_id=FRIEND, session_id="one", sequence=1),
            self.message(sender=ROBOT, sequence=2),
            Message("我自己说", FRIEND, sender_id=FRIEND, is_self=True, session_id="one", sequence=3),
        )
        for message in ignored:
            self.store.record(message)
            self.plugin.on_message(message, self.ctx)
        self.ctx.reply.assert_not_called()
        heard = self.message(sequence=4)
        self.store.record(heard)
        self.plugin.on_message(heard, self.ctx)
        self.assertEqual(self.ctx.reply.call_args.args[:2], (heard, "收到，这是私聊回答"))
        tools = [tool.name for tool in self.llm.chat.call_args.args[2]]
        self.assertEqual(tools, ["view_image", "search_history"])
        body = json.loads(self.llm.chat.call_args.args[1])
        self.assertEqual(body["diary"], [])
        self.assertEqual(body["memory"], [])
        self.assertEqual(self.store.search_room_lines(FRIEND, "私聊回答", 1)[-1]["text"], "收到，这是私聊回答")

    def test_model_failure_sends_a_short_retry_notice(self):
        self.llm.chat.side_effect = LLMError("失败")
        heard = self.message(sequence=8)
        self.store.record(heard)
        self.plugin.on_message(heard, self.ctx)
        self.assertEqual(self.ctx.reply.call_args.args[:2], (heard, "暂时无法回复，请稍后再试。"))

    def test_disabled_history_refuses_to_load(self):
        self.path.write_text(json.dumps({
            "database_file": "data.sqlite3", "history_enabled": False, "groups": {ROOM: []},
        }, ensure_ascii=False), encoding="utf-8")
        disabled = MemberStore(self.path, ROBOT)
        self.addCleanup(disabled.close)
        ctx = Mock(members=disabled, llm=self.llm)
        with self.assertRaisesRegex(ValueError, "history_enabled"):
            Plugin().on_load(ctx)


class PrivatePanelTests(PanelFixture):
    def test_save_updates_the_running_list_and_extends_a_nonempty_allowlist(self):
        path = self.root / "bot_config.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["wechat_account"] = "wxid_test_robot"
        data["chat_allowlist"] = []
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.controller.start()
        store = MemberStore(self.root / "members_config.json", "wxid_test_robot")
        self.addCleanup(store.close)
        self.controller.app.members = store
        payload = {"contacts": [FRIEND], "names": {FRIEND: "朋友"},
                   "context_hours": 12, "context_limit": 40, "personality": "默认"}
        saved = self.controller.save_private_chat(payload)
        self.assertEqual(saved["private_chat"]["contacts"], [FRIEND])
        self.assertEqual(saved["private_chat"]["context_hours"], 12)
        self.assertEqual(self.controller.app.config["chat_allowlist"], [])
        data["chat_allowlist"] = [ROOM]
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.controller.save_private_chat(payload)
        self.assertEqual(self.controller.app.config["chat_allowlist"], [ROOM, FRIEND])
        self.assertIn(FRIEND, store.private_contacts)
        self.wait_stopped()
