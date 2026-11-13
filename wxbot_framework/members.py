"""框架共用的成员身份和新收到的群文本记录，按账号、群及成员 ID 隔离。"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .account import validate_account
from .config import validate_name
from .memory import DIARY_DAYS, SOURCE_CHARS, SOURCE_LINE_CHARS, SOURCE_LINES
from .room_behavior import (BehaviorError, RESERVED_PERSONALITY, alias_mentioned as name_mentioned,
                            normalize_personalities, normalize_room_setting, public_personalities,
                            normalize_global_prompt, normalize_reply_limit, write_json)
from .text import validate_mentions

logger = logging.getLogger("Members")


class MemberError(ValueError):
    pass


class SnapshotExpired(RuntimeError):
    """清空或跨天后，取消仍在生成/等待发送的总结。"""


BEIJING = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class ChatSnapshot:
    member: Member
    day: str
    generation: int
    messages: tuple

    @property
    def room(self):
        return self.member.room


@dataclass(frozen=True)
class GroupChatSnapshot:
    room: str
    day: str
    generation: int
    messages: tuple


@dataclass(frozen=True)
class ConversationSnapshot:
    room: str
    day: str
    generation: int
    speaker_name: str
    messages: tuple
    current: str


@dataclass(frozen=True)
class Member:
    room: str
    name: str
    display_name: str
    wxid: str | None
    aliases: tuple[str, ...]
    self_named: bool = False


@dataclass(frozen=True)
class VoiceSelection:
    room: str
    voice_id: str = ""
    name: str = ""
    enabled: bool = True
    generation: int = 0
    mode: str = "always"


def name_key(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 64 or any(c in value for c in "\0\n\r"):
        raise MemberError("成员称呼应为 1–64 字符的单行文本")
    return value.strip().casefold()


class MemberStore:
    def __init__(self, config_file, account, *, now=None):
        self.account = validate_account(account)
        self.config_file = Path(config_file).resolve()
        options = json.loads(self.config_file.read_text(encoding="utf-8-sig"))
        if not isinstance(options, dict) or not isinstance(options.get("groups"), dict):
            raise MemberError("成员配置应包含 groups 对象")
        database = options.get("database_file", ".runtime/member_data.sqlite3")
        if not isinstance(database, str) or not database.strip() or "\0" in database:
            raise MemberError("database_file 应为数据库路径")
        self.path = (self.config_file.parent / database).resolve()
        self.history_enabled = options.get("history_enabled", False)
        if type(self.history_enabled) is not bool:
            raise MemberError("history_enabled 必须是布尔值")
        self.daily_clear = options.get("daily_clear", False)
        if type(self.daily_clear) is not bool:
            raise MemberError("daily_clear 必须是布尔值")
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._active_day = None
        self.groups = set(options["groups"])
        raw_names = options.get("group_names") or {}
        self.group_names = {}
        if isinstance(raw_names, dict):
            for room, title in raw_names.items():
                if (isinstance(room, str) and isinstance(title, str) and title.strip() and
                        len(title) <= 64 and not any(ord(ch) < 32 for ch in title)):
                    self.group_names[room] = title.strip()
        self.profiles = []
        for room, profiles in options["groups"].items():
            validate_mentions(room, [self.account])
            if not isinstance(profiles, list):
                raise MemberError("每个群的成员配置应为列表")
            keys, names, ids = set(), set(), set()
            for profile in profiles:
                if not isinstance(profile, dict):
                    raise MemberError("成员资料应为 JSON 对象")
                key = validate_name(profile.get("key"))
                name = profile.get("name")
                display = profile.get("display_name", name)
                aliases = profile.get("aliases", [])
                if not isinstance(aliases, list):
                    raise MemberError("aliases 应为称呼列表")
                normalized = {name_key(value) for value in [name, display, *aliases]}
                wxid = profile.get("wxid")
                if wxid is not None:
                    validate_mentions(room, [wxid])
                if key in keys or normalized & names or (wxid and wxid in ids):
                    raise MemberError("同一群的成员标识、称呼和微信 ID 不能重复")
                keys.add(key)
                names.update(normalized)
                if wxid:
                    ids.add(wxid)
                self.profiles.append((room, key, name.strip(), display.strip(), wxid, normalized))
        self._load_behavior(options)
        self._lock = threading.RLock()
        self._db = None
        self._closed = False

    def _load_behavior(self, options):
        try:
            items = normalize_personalities(options["personalities"] if "personalities" in options else None)
        except BehaviorError as error:
            raise MemberError(str(error)) from error
        self.personalities = public_personalities(items)
        try:
            self.global_prompt = normalize_global_prompt(options["global_prompt"] if "global_prompt" in options else None)
            self.reply_limit = normalize_reply_limit(options.get("reply_limit"))
        except BehaviorError as error:
            raise MemberError(str(error)) from error
        raw_rooms = options.get("room_settings") or {}
        if not isinstance(raw_rooms, dict):
            raise MemberError("room_settings 应为对象")
        names = [item["name"] for item in self.personalities]
        self.room_settings = {}
        for room, value in raw_rooms.items():
            if room not in self.groups:
                continue
            try:
                self.room_settings[room] = normalize_room_setting(value, names)
            except (BehaviorError, ValueError) as error:
                raise MemberError(str(error)) from error

    @classmethod
    def from_config(cls, config, directory):
        filename = config.get("members_config_file")
        if filename is None:
            return None
        if not isinstance(filename, str) or not filename.strip() or "\0" in filename:
            raise MemberError("members_config_file 应为成员配置文件路径或 null")
        if not config.get("wechat_account"):
            raise MemberError("启用成员目录需要固定 wechat_account，以隔离各机器人账号的数据")
        return cls(Path(directory) / filename, config["wechat_account"])

    def _connection(self):
        if self._closed:
            raise RuntimeError("成员目录已关闭")
        if self._db is not None:
            return self._db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5, check_same_thread=False)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS people (
                    account TEXT NOT NULL, room TEXT NOT NULL, person_key TEXT NOT NULL,
                    name TEXT NOT NULL, display_name TEXT NOT NULL, wxid TEXT,
                    PRIMARY KEY(account, room, person_key), UNIQUE(account, room, wxid));
                CREATE TABLE IF NOT EXISTS aliases (
                    account TEXT NOT NULL, room TEXT NOT NULL, alias TEXT NOT NULL, person_key TEXT NOT NULL,
                    PRIMARY KEY(account, room, alias),
                    FOREIGN KEY(account, room, person_key) REFERENCES people(account, room, person_key));
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, account TEXT NOT NULL, room TEXT NOT NULL, sender_id TEXT NOT NULL,
                    text TEXT NOT NULL, received_at TEXT NOT NULL, session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL, is_self INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS messages_by_member ON messages(account, room, sender_id, id);
                CREATE UNIQUE INDEX IF NOT EXISTS messages_by_sequence
                    ON messages(account, session_id, sequence) WHERE sequence > 0;
                CREATE TABLE IF NOT EXISTS chat_buffers (
                    account TEXT NOT NULL, room TEXT NOT NULL, local_day TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(account,room));
                CREATE TABLE IF NOT EXISTS room_personalities (
                    account TEXT NOT NULL, room TEXT NOT NULL, name TEXT NOT NULL,
                    generation INTEGER NOT NULL, PRIMARY KEY(account,room));
                CREATE TABLE IF NOT EXISTS wechat_names (
                    account TEXT NOT NULL, wxid TEXT NOT NULL, name TEXT NOT NULL, name_key TEXT NOT NULL,
                    PRIMARY KEY(account,wxid));
                CREATE TABLE IF NOT EXISTS room_members (
                    account TEXT NOT NULL, room TEXT NOT NULL, wxid TEXT NOT NULL,
                    PRIMARY KEY(account,room,wxid));
                CREATE TABLE IF NOT EXISTS group_nicknames (
                    account TEXT NOT NULL, room TEXT NOT NULL, wxid TEXT NOT NULL, nickname TEXT NOT NULL,
                    PRIMARY KEY(account,room,wxid));
                CREATE TABLE IF NOT EXISTS room_voices (
                    account TEXT NOT NULL, room TEXT NOT NULL, voice_id TEXT NOT NULL,
                    name TEXT NOT NULL, enabled INTEGER NOT NULL, generation INTEGER NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'always', PRIMARY KEY(account,room));
                CREATE TABLE IF NOT EXISTS room_digests (
                    account TEXT NOT NULL, room TEXT NOT NULL, local_day TEXT NOT NULL,
                    summary TEXT NOT NULL, PRIMARY KEY(account,room,local_day));
                CREATE TABLE IF NOT EXISTS pending_digests (
                    account TEXT NOT NULL, room TEXT NOT NULL, local_day TEXT NOT NULL,
                    source TEXT NOT NULL, PRIMARY KEY(account,room,local_day));
                CREATE TABLE IF NOT EXISTS person_facts (
                    account TEXT NOT NULL, room TEXT NOT NULL, sender_id TEXT NOT NULL,
                    facts TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(account,room,sender_id));
                CREATE TABLE IF NOT EXISTS room_proactive (
                    account TEXT NOT NULL, room TEXT NOT NULL, bias REAL NOT NULL,
                    PRIMARY KEY(account,room));
                INSERT OR IGNORE INTO room_members SELECT DISTINCT account,room,sender_id FROM messages;
            """)
            if "self_named" not in {row["name"] for row in db.execute("PRAGMA table_info(people)")}:
                db.execute("ALTER TABLE people ADD COLUMN self_named INTEGER NOT NULL DEFAULT 0")
            if "bot_generated" not in {row["name"] for row in db.execute("PRAGMA table_info(messages)")}:
                db.execute("ALTER TABLE messages ADD COLUMN bot_generated INTEGER NOT NULL DEFAULT 0")
            if "mode" not in {row["name"] for row in db.execute("PRAGMA table_info(room_voices)")}:
                db.execute("ALTER TABLE room_voices ADD COLUMN mode TEXT NOT NULL DEFAULT 'always'")
            with db:
                for room, key, name, display, wxid, aliases in self.profiles:
                    existing = db.execute("SELECT wxid FROM people WHERE account=? AND room=? AND person_key=?",
                                          (self.account, room, key)).fetchone()
                    if existing and existing["wxid"] and wxid and existing["wxid"] != wxid:
                        raise MemberError(f"{name} 的配置 ID 与已保存绑定不同，请先核对")
                    db.execute("""INSERT INTO people(account,room,person_key,name,display_name,wxid) VALUES(?,?,?,?,?,?)
                        ON CONFLICT(account,room,person_key) DO UPDATE SET
                            name=CASE WHEN people.self_named=0 THEN excluded.name ELSE people.name END,
                            display_name=CASE WHEN people.self_named=0 THEN excluded.display_name ELSE people.display_name END,
                            wxid=COALESCE(people.wxid,excluded.wxid)""", (self.account, room, key, name, display, wxid))
                    for alias in aliases:
                        old = db.execute("SELECT person_key FROM aliases WHERE account=? AND room=? AND alias=?",
                                         (self.account, room, alias)).fetchone()
                        if old and old["person_key"] != key:
                            raise MemberError(f"称呼 {alias} 已属于另一位成员")
                        db.execute("INSERT OR IGNORE INTO aliases VALUES(?,?,?,?)", (self.account, room, alias, key))
            self._db = db
            return db
        except BaseException:
            db.close()
            raise

    def initialize(self):
        with self._lock:
            self._connection()
            self.rollover()

    def _day_bounds(self):
        local = self._now().astimezone(BEIJING)
        start = local.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start.date().isoformat(), start.astimezone(timezone.utc).isoformat(),
                (start + timedelta(days=1)).astimezone(timezone.utc).isoformat())

    def rollover(self):
        """即使没有新消息也可调用；停机期间跨天则在启动时补清理。"""
        with self._lock:
            day, start, _ = self._day_bounds()
            if day == self._active_day:
                return False
            with self._connection() as db:
                for room in self.groups:
                    db.execute("""INSERT INTO chat_buffers(account,room,local_day) VALUES(?,?,?)
                        ON CONFLICT(account,room) DO UPDATE SET local_day=excluded.local_day,
                        generation=chat_buffers.generation+1 WHERE chat_buffers.local_day<>excluded.local_day""",
                               (self.account, room, day))
                    if self.daily_clear:
                        # 先留下可压缩的材料，再删原文。日记写好之前，回复不会读这段原文。
                        self._stash_digest_source(db, room, start)
                        db.execute("DELETE FROM messages WHERE account=? AND room=? AND received_at<?",
                                   (self.account, room, start))
            self._active_day = day
            return True

    def clear(self, room):
        if room not in self.groups:
            raise MemberError("这个群尚未启用聊天记录")
        with self._lock:
            self.rollover()
            with self._connection() as db:
                count = db.execute("DELETE FROM messages WHERE account=? AND room=?", (self.account, room)).rowcount
                db.execute("DELETE FROM room_digests WHERE account=? AND room=?", (self.account, room))
                db.execute("DELETE FROM pending_digests WHERE account=? AND room=?", (self.account, room))
                db.execute("DELETE FROM person_facts WHERE account=? AND room=?", (self.account, room))
                db.execute("UPDATE chat_buffers SET generation=generation+1 WHERE account=? AND room=?",
                           (self.account, room))
            return count

    def _stash_digest_source(self, db, room, start):
        """把即将删掉的文字收成待压缩日记。同一天已有材料或日记时不重复收。"""
        rows = db.execute("""SELECT sender_id, text, bot_generated, received_at FROM messages
            WHERE account=? AND room=? AND received_at<? ORDER BY id DESC LIMIT ?""",
            (self.account, room, start, SOURCE_LINES)).fetchall()
        if not rows:
            return
        local_day = datetime.fromisoformat(rows[0]["received_at"]).astimezone(BEIJING).date().isoformat()
        exists = db.execute("""SELECT 1 FROM room_digests WHERE account=? AND room=? AND local_day=?
            UNION SELECT 1 FROM pending_digests WHERE account=? AND room=? AND local_day=?""",
            (self.account, room, local_day, self.account, room, local_day)).fetchone()
        if exists:
            return
        source = self._compact_source(room, rows)
        if source:
            db.execute("INSERT INTO pending_digests(account,room,local_day,source) VALUES(?,?,?,?)",
                       (self.account, room, local_day, source))

    def _compact_source(self, room, rows):
        """从新到旧截到字数上限，再按时间排回去。"""
        from .summary import CONTROL
        lines, size = [], 0
        for row in rows:
            text = " ".join(str(row["text"]).split())
            if not text or "WXBOT_AT_CHAT_PROBE" in text or text.split(maxsplit=1)[0] in CONTROL:
                continue
            speaker = self._context_speaker(room, row["sender_id"], row["bot_generated"])
            moment = datetime.fromisoformat(row["received_at"]).astimezone(BEIJING).strftime("%H:%M")
            line = f"{moment} {speaker}：{text[:SOURCE_LINE_CHARS]}"
            if lines and size + len(line) + 1 > SOURCE_CHARS:
                break
            lines.append(line)
            size += len(line) + 1
        lines.reverse()
        return "\n".join(lines)

    def pending_digests(self):
        """还没写成日记的材料。某一天压缩失败时，其他天仍可以继续。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("成员目录已关闭")
            rows = self._connection().execute("""SELECT room, local_day, source FROM pending_digests
                WHERE account=? ORDER BY local_day""", (self.account,)).fetchall()
            return tuple((row["room"], row["local_day"], row["source"]) for row in rows)

    def finish_digest(self, room, local_day, summary):
        """材料还在才写入日记。清空已经删掉材料时，不再把日记写回去。"""
        with self._lock, self._connection() as db:
            removed = db.execute("DELETE FROM pending_digests WHERE account=? AND room=? AND local_day=?",
                                 (self.account, room, local_day)).rowcount
            if not removed:
                return False
            db.execute("""INSERT INTO room_digests(account,room,local_day,summary) VALUES(?,?,?,?)
                ON CONFLICT(account,room,local_day) DO UPDATE SET summary=excluded.summary""",
                       (self.account, room, local_day, summary))
            return True

    def recent_digests(self, room, limit=DIARY_DAYS):
        with self._lock:
            rows = self._connection().execute("""SELECT local_day, summary FROM room_digests
                WHERE account=? AND room=? ORDER BY local_day DESC LIMIT ?""",
                (self.account, room, limit)).fetchall()
        return tuple({"date": row["local_day"], "summary": row["summary"]} for row in reversed(rows))

    def facts_for(self, room, sender_id):
        with self._lock:
            row = self._connection().execute("""SELECT facts FROM person_facts
                WHERE account=? AND room=? AND sender_id=?""",
                (self.account, room, sender_id)).fetchone()
            if row is None:
                return ()
            try:
                data = json.loads(row[0])
            except json.JSONDecodeError:
                return ()
            if not isinstance(data, list):
                return ()
            return tuple(item for item in data if isinstance(item, str) and item.strip())

    def replace_facts(self, room, sender_id, facts, generation):
        """清空或跨天之后，不再写回这次对话开始前看到的档案。"""
        payload = json.dumps(list(facts), ensure_ascii=False)
        with self._lock, self._connection() as db:
            state = db.execute("SELECT generation FROM chat_buffers WHERE account=? AND room=?",
                               (self.account, room)).fetchone()
            if state is None or state[0] != generation:
                return False
            db.execute("""INSERT INTO person_facts(account,room,sender_id,facts,updated_at) VALUES(?,?,?,?,?)
                ON CONFLICT(account,room,sender_id) DO UPDATE SET facts=excluded.facts,
                updated_at=excluded.updated_at""",
                (self.account, room, sender_id, payload, self._now().astimezone(timezone.utc).isoformat()))
            return True

    def today_snapshot(self, name, room):
        """完整读取当天的成员文本，数量不受控制台 history 的条数限制。"""
        with self._lock:
            self.rollover()
            member = self.resolve(name, room)
            if member.wxid is None:
                raise MemberError(f"{member.name} 尚未绑定微信 ID")
            day, start, end = self._day_bounds()
            db = self._connection()
            state = db.execute("SELECT generation FROM chat_buffers WHERE account=? AND room=?",
                               (self.account, room)).fetchone()
            if state is None:
                raise MemberError("这个群尚未启用聊天记录")
            rows = db.execute("""SELECT * FROM messages WHERE account=? AND room=? AND sender_id=?
                AND received_at>=? AND received_at<? AND bot_generated=0 ORDER BY id""",
                              (self.account, room, member.wxid, start, end)).fetchall()
            return ChatSnapshot(member, day, state[0], tuple(dict(row) for row in rows))

    def today_group_snapshot(self, room):
        """完整读取本群今天所有成员的文本，未绑定者也保留身份区分。"""
        if room not in self.groups:
            raise MemberError("这个群尚未启用聊天记录")
        with self._lock:
            self.rollover()
            day, start, end = self._day_bounds()
            db = self._connection()
            state = db.execute("SELECT generation FROM chat_buffers WHERE account=? AND room=?",
                               (self.account, room)).fetchone()
            rows = db.execute("""SELECT m.*, p.name AS speaker_name FROM messages m
                LEFT JOIN people p ON p.account=m.account AND p.room=m.room AND p.wxid=m.sender_id
                WHERE m.account=? AND m.room=? AND m.received_at>=? AND m.received_at<?
                AND m.bot_generated=0 ORDER BY m.id""", (self.account, room, start, end)).fetchall()
            speakers, messages = {}, []
            for row in rows:
                item = dict(row)
                number = speakers.setdefault(item["sender_id"], len(speakers) + 1)
                # 发给模型的稳定编号替代真实微信 ID；昵称只用于可读性。
                item["speaker"] = f"P{number}"
                item["speaker_name"] = item["speaker_name"] or self.display_name(item["sender_id"], room) or f"未绑定成员{number}"
                messages.append(item)
            return GroupChatSnapshot(room, day, state[0], tuple(messages))

    def snapshot_is_current(self, snapshot):
        with self._lock:
            self.rollover()
            state = self._connection().execute("SELECT local_day,generation FROM chat_buffers WHERE account=? AND room=?",
                                               (self.account, snapshot.room)).fetchone()
            return bool(state and state[0] == snapshot.day and state[1] == snapshot.generation)

    def personality(self, room):
        from .personality import PersonalityState
        with self._lock:
            row = self._connection().execute("SELECT name,generation FROM room_personalities WHERE account=? AND room=?",
                                              (self.account, room)).fetchone()
            return PersonalityState(room, row[0], row[1]) if row else PersonalityState(room)

    def personality_names(self):
        return tuple(item["name"] for item in self.personalities)

    def prompts(self):
        return {item["name"]: item["prompt"] for item in self.personalities}

    def room_admins(self, room):
        with self._lock:
            return tuple((self.room_settings.get(room) or {}).get("admins", ()))

    def room_aliases(self, room):
        with self._lock:
            return tuple((self.room_settings.get(room) or {}).get("aliases", ()))

    def disabled_plugins(self, room):
        with self._lock:
            values = (self.room_settings.get(room) or {}).get("disabled_plugins") or ()
            return frozenset(value for value in values if isinstance(value, str))

    def set_room_plugin_enabled(self, room, name, enabled):
        """按群打开或关闭一个功能，并立刻写入群设置。"""
        from .config import validate_name
        plugin = validate_name(name)
        if room not in self.groups or type(enabled) is not bool:
            raise MemberError("请在已配置的群中开关功能")
        with self._lock:
            current = dict(self.room_settings.get(room) or {})
            disabled = [item for item in current.get("disabled_plugins") or [] if item != plugin]
            if not enabled:
                disabled.append(plugin)
            current["disabled_plugins"] = disabled
            self.apply_room(room, current)
            return plugin not in self.disabled_plugins(room)

    def plugin_options(self, room, name):
        """这个群单独改过的插件选项。没改过的项不在这里，调用方继续用账号配置。"""
        with self._lock:
            options = (self.room_settings.get(room) or {}).get("plugin_options") or {}
            if not isinstance(options, dict):
                return {}
            value = options.get(name)
            return dict(value) if isinstance(value, dict) else {}

    def alias_mentioned(self, room, text):
        return name_mentioned(text, self.room_aliases(room))

    def set_personality(self, room, name, *, mirror=True):
        allowed = self.personality_names()
        if room not in self.groups or (name not in allowed and name != RESERVED_PERSONALITY):
            label = "、".join((*allowed, RESERVED_PERSONALITY))
            raise MemberError(f"人格应为 {label}，且在已启用的群中切换")
        with self._lock:
            current = self.personality(room)
            if current.name == name:
                return current
            with self._connection() as db:
                db.execute("""INSERT INTO room_personalities VALUES(?,?,?,1)
                    ON CONFLICT(account,room) DO UPDATE SET name=excluded.name,
                    generation=room_personalities.generation+1 WHERE room_personalities.name<>excluded.name""",
                           (self.account, room, name))
            result = self.personality(room)
            if mirror:
                self._mirror_personality(room, result.name)
            return result

    def update_behavior(self, data):
        """保存人格库或单个群的管理员、别名和启用人格，并立刻作用于当前进程。"""
        if not isinstance(data, dict) or ("personalities" not in data and "room" not in data
                                          and "global_prompt" not in data and "reply_limit" not in data):
            raise MemberError("没有要保存的群聊行为")
        if "reply_limit" in data:
            self.reply_limit = normalize_reply_limit(data["reply_limit"])
        if "global_prompt" in data:
            self.global_prompt = normalize_global_prompt(data["global_prompt"])
        if "personalities" in data:
            self.apply_personalities(data["personalities"])
        elif "global_prompt" in data or "reply_limit" in data:
            with self._lock:
                self._persist_behavior()
        if "room" in data:
            self.apply_room(data["room"], data.get("setting"))

    def apply_personalities(self, raw):
        items = normalize_personalities(raw)
        with self._lock:
            old = {item["name"]: item["prompt"] for item in self.personalities}
            for item in items:
                previous = item["previous"]
                source = previous or item["name"]
                if previous and previous != item["name"]:
                    if previous not in old:
                        raise MemberError(f"找不到要改名的人格：{previous}")
                    self._rename_personality(previous, item["name"])
                if old.get(source) not in (None, item["prompt"]):
                    self._bump_personality(item["name"])
            new_names = {item["name"] for item in items}
            for room in self.groups:
                current = self.personality(room).name
                if current not in new_names and current != RESERVED_PERSONALITY:
                    self.set_personality(room, RESERVED_PERSONALITY, mirror=False)
                    setting = self.room_settings.get(room)
                    if setting is not None:
                        setting["personality"] = RESERVED_PERSONALITY
            self.personalities = public_personalities(items)
            self._persist_behavior()

    def apply_room(self, room, setting):
        if room not in self.groups:
            raise MemberError("请先把这个群加入支持的群聊")
        cleaned = normalize_room_setting(setting, self.personality_names())
        with self._lock:
            previous = self.room_settings.get(room)
            self.room_settings[room] = cleaned
            try:
                self._persist_room_settings()
            except Exception:
                if previous is None:
                    self.room_settings.pop(room, None)
                else:
                    self.room_settings[room] = previous
                raise
            if self.personality(room).name != cleaned["personality"]:
                self.set_personality(room, cleaned["personality"], mirror=False)

    def _apply_saved_personalities(self):
        names = set(self.personality_names())
        for room in self.groups:
            saved = (self.room_settings.get(room) or {}).get("personality")
            current = self.personality(room).name
            target = current if saved is None else saved
            if target not in names and target != RESERVED_PERSONALITY:
                target = RESERVED_PERSONALITY
            if current != target:
                self.set_personality(room, target, mirror=True)

    def _rename_personality(self, source, target):
        with self._connection() as db:
            db.execute("""UPDATE room_personalities SET name=?, generation=generation+1
                          WHERE account=? AND name=?""", (target, self.account, source))
        for setting in self.room_settings.values():
            if setting.get("personality") == source:
                setting["personality"] = target

    def _bump_personality(self, name):
        with self._connection() as db:
            db.execute("""UPDATE room_personalities SET generation=generation+1
                          WHERE account=? AND name=?""", (self.account, name))

    def _mirror_personality(self, room, name):
        current = dict(self.room_settings.get(room) or {"admins": [], "aliases": [], "personality": RESERVED_PERSONALITY})
        current["personality"] = name
        self.room_settings[room] = current
        try:
            self._persist_room_settings()
        except (OSError, json.JSONDecodeError):
            logger.warning("人格选择未能写入配置：room=%s", room, exc_info=True)

    def _persist_room_settings(self):
        self._mutate_config(lambda raw: raw.__setitem__("room_settings", self.room_settings))

    def _persist_behavior(self):
        def mutate(raw):
            raw["personalities"] = self.personalities
            raw["global_prompt"] = self.global_prompt
            raw["reply_limit"] = self.reply_limit
            raw["room_settings"] = self.room_settings
        self._mutate_config(mutate)

    def _mutate_config(self, mutate):
        raw = json.loads(self.config_file.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise MemberError("成员配置应包含 groups 对象")
        mutate(raw)
        write_json(self.config_file, raw)

    def personality_is_current(self, state):
        return self.personality(state.room) == state

    def voice_selection(self, room):
        with self._lock:
            row = self._connection().execute(
                "SELECT voice_id,name,enabled,generation,mode FROM room_voices WHERE account=? AND room=?",
                (self.account, room)).fetchone()
            if not row:
                return VoiceSelection(room)
            mode = row["mode"] if row["mode"] in ("always", "auto") else "always"
            return VoiceSelection(room, row["voice_id"], row["name"], bool(row["enabled"]), row["generation"], mode)

    def set_voice(self, room, voice_id, name, *, enabled=True, mode=None):
        from .tts import VOICE_ID
        if room not in self.groups or type(enabled) is not bool:
            raise MemberError("请在已配置的群中切换音色")
        if enabled and (not isinstance(voice_id, str) or not VOICE_ID.fullmatch(voice_id)):
            raise MemberError("音色 ID 无效")
        if not isinstance(name, str) or len(name) > 128 or any(ord(ch) < 32 for ch in name):
            raise MemberError("音色名称无效")
        if mode is not None and mode not in ("always", "auto"):
            raise MemberError("语音方式应为始终或自动")
        with self._lock, self._connection() as db:
            current = self.voice_selection(room)
            if not enabled:
                voice_id, name = current.voice_id, current.name
                mode = current.mode
            elif mode is None:
                mode = current.mode
            if (current.voice_id, current.name, current.enabled, current.mode) != (voice_id, name, enabled, mode):
                db.execute("""INSERT INTO room_voices(account,room,voice_id,name,enabled,generation,mode)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(account,room) DO UPDATE SET voice_id=excluded.voice_id,
                    name=excluded.name,enabled=excluded.enabled,generation=excluded.generation,mode=excluded.mode""",
                    (self.account, room, voice_id, name, int(enabled), current.generation + 1, mode))
            return self.voice_selection(room)

    def voice_modes(self):
        """各群当前音色和语音方式，给管理台。没选过的群不在这里。"""
        with self._lock:
            rows = self._connection().execute(
                "SELECT room, voice_id, name, enabled, mode FROM room_voices WHERE account=?",
                (self.account,)).fetchall()
        result = {}
        for row in rows:
            result[row["room"]] = {"voice_id": row["voice_id"], "name": row["name"],
                                   "enabled": bool(row["enabled"]),
                                   "mode": row["mode"] if row["mode"] in ("always", "auto") else "always"}
        return result

    def assign_room_voice(self, room, mode, voice_id=""):
        """按页面或指令设置本群音色和语音方式。voice_id 为空时沿用本群已经保存的 ID。"""
        from .tts import VOICE_ID, normalize_voice_id
        supplied = normalize_voice_id(voice_id)
        if supplied and not VOICE_ID.fullmatch(supplied):
            raise MemberError("音色 ID 无效。Fish 是 32 位十六进制，Breeze 以 voc_ 开头")
        current = self.voice_selection(room)
        if mode == "off":
            if supplied:
                name = current.name if current.voice_id == supplied and current.name else supplied
                kept = current.mode if current.voice_id else "always"
                self.set_voice(room, supplied, name, enabled=True, mode=kept)
            return self.set_voice(room, "", "", enabled=False)
        if mode not in ("always", "auto"):
            raise MemberError("语音回复应为关闭、始终语音或自动")
        chosen = supplied or current.voice_id
        if not chosen:
            raise MemberError("请填写本群音色 ID，或在「模型与语音」中填写默认音色 ID")
        if not VOICE_ID.fullmatch(chosen):
            raise MemberError("音色 ID 无效。Fish 是 32 位十六进制，Breeze 以 voc_ 开头")
        name = current.name if current.voice_id == chosen and current.name else chosen
        return self.set_voice(room, chosen, name, enabled=True, mode=mode)

    def set_voice_mode(self, room, mode):
        """关闭、始终语音或自动。后两种沿用本群已经保存的音色 ID。"""
        return self.assign_room_voice(room, mode)

    @contextmanager
    def voice_delivery_guard(self, state):
        with self._lock:
            if self.voice_selection(state.room) != state:
                raise SnapshotExpired("音色已切换，取消旧语音")
            yield

    @contextmanager
    def personality_delivery_guard(self, state):
        from .personality import PersonalityExpired
        with self._lock:
            if not self.personality_is_current(state):
                raise PersonalityExpired("人格已切换")
            yield

    def context_window(self, room):
        """这个群从库里读取的时间范围和条数。没单独配置时是 6 小时、300 条。送给模型时还会再收成最近一小段。"""
        from .room_behavior import DEFAULT_CONTEXT_HOURS, DEFAULT_CONTEXT_LIMIT
        setting = self.room_settings.get(room) or {}
        return (setting.get("context_hours", DEFAULT_CONTEXT_HOURS),
                setting.get("context_limit", DEFAULT_CONTEXT_LIMIT))

    def room_unnamed_reply(self, room):
        """本群是否开启未点名回应。没写时看旧的活跃度和主动插话开关。"""
        from .room_behavior import BehaviorError, normalize_unnamed_reply
        setting = self.room_settings.get(room) or {}
        try:
            return normalize_unnamed_reply(setting.get("unnamed_reply"), setting)
        except BehaviorError:
            return False

    def room_hourly_limit(self, room):
        """未点名回应每小时按节拍最多说几句。"""
        from .room_behavior import BehaviorError, DEFAULT_HOURLY_LIMIT, normalize_hourly_limit
        setting = self.room_settings.get(room) or {}
        try:
            return normalize_hourly_limit(setting.get("hourly_limit"))
        except BehaviorError:
            return DEFAULT_HOURLY_LIMIT

    def room_speak_threshold(self, room):
        """未点名回应时，决策概率达到这个数才说话。没写时是 0.5。"""
        from .room_behavior import BehaviorError, DEFAULT_SPEAK_THRESHOLD, normalize_speak_threshold
        setting = self.room_settings.get(room) or {}
        try:
            return normalize_speak_threshold(setting.get("speak_threshold"))
        except BehaviorError:
            return DEFAULT_SPEAK_THRESHOLD

    def room_voice_threshold(self, room):
        """自动语音时，决策概率达到这个数才念出来。没写时是 0.6。"""
        from .room_behavior import BehaviorError, DEFAULT_VOICE_THRESHOLD, normalize_voice_threshold
        setting = self.room_settings.get(room) or {}
        try:
            return normalize_voice_threshold(setting.get("voice_threshold"), setting)
        except BehaviorError:
            return DEFAULT_VOICE_THRESHOLD

    def room_glance_minutes(self, room):
        """未点名回应隔几分钟看一眼群，0 表示不按节拍看。"""
        from .room_behavior import BehaviorError, DEFAULT_GLANCE_MINUTES, normalize_glance_minutes
        setting = self.room_settings.get(room) or {}
        try:
            return normalize_glance_minutes(setting.get("glance_minutes"))
        except BehaviorError:
            return DEFAULT_GLANCE_MINUTES

    def room_deep_thinking(self, room):
        """本群是否让 DeepSeek 先思考。没写过时开启。"""
        from .room_behavior import BehaviorError, normalize_deep_thinking
        setting = self.room_settings.get(room) or {}
        try:
            return normalize_deep_thinking(setting.get("deep_thinking"))
        except BehaviorError:
            return True

    def bot_said(self, room, text, limit=50):
        """被引用的文字是不是机器人最近在本群说过的话。微信引用里不带作者 ID，只能比对正文。"""
        quoted = " ".join(str(text or "").split()).rstrip("…. ")
        if len(quoted) < 2:
            return False
        with self._lock:
            rows = self._connection().execute(
                """SELECT text FROM messages WHERE account=? AND room=? AND bot_generated=1
                   ORDER BY id DESC LIMIT ?""", (self.account, room, limit)).fetchall()
        head = quoted[:30]
        for row in rows:
            said = " ".join(row["text"].split())
            if said == quoted or (len(head) >= 4 and said.startswith(head)):
                return True
        return False

    def _context_speaker(self, room, sender_id, bot_generated):
        """流水里标出发言人。机器人自己的回复用本群别名，没有别名就标成机器人。"""
        if bot_generated or sender_id == self.account:
            aliases = (self.room_settings.get(room) or {}).get("aliases") or ()
            return aliases[0] if aliases else "机器人"
        return self.display_name(sender_id, room) or "群友"

    def conversation_snapshot(self, message):
        """本群最近一段时间内、当前消息之前的文字。后来的消息不带入。"""
        from .summary import CONTROL
        if message.chat_id not in self.groups:
            raise MemberError("这个群尚未启用聊天记录")
        with self._lock:
            self.rollover()
            day, _, _ = self._day_bounds()
            hours, limit = self.context_window(message.chat_id)
            since = (self._now() - timedelta(hours=hours)).astimezone(timezone.utc).isoformat()
            db = self._connection()
            current = db.execute("""SELECT id FROM messages WHERE account=? AND room=? AND sender_id=?
                AND session_id=? AND sequence=? AND text=? AND received_at>=?
                ORDER BY id DESC LIMIT 1""", (self.account, message.chat_id, message.sender_id,
                message.session_id, message.sequence, message.text, since)).fetchone()
            if current is None:
                raise SnapshotExpired("当前消息已被清空或已超出上下文时间")
            cursor = db.execute("""SELECT * FROM messages WHERE account=? AND room=? AND id<?
                AND received_at>=? ORDER BY id DESC""",
                (self.account, message.chat_id, current[0], since))
            previous, budget = [], max(limit - 1, 0)
            for row in cursor:
                if len(previous) >= budget:
                    break
                text = row["text"].strip()
                if text and "WXBOT_AT_CHAT_PROBE" not in text and text.split(maxsplit=1)[0] not in CONTROL:
                    item = dict(row)
                    item["speaker"] = self._context_speaker(message.chat_id, row["sender_id"], row["bot_generated"])
                    previous.append(item)
            member = self.find_by_wxid(message.sender_id, message.chat_id)
            state = db.execute("SELECT generation FROM chat_buffers WHERE account=? AND room=?",
                               (self.account, message.chat_id)).fetchone()
            return ConversationSnapshot(message.chat_id, day, state[0], member.name if member else
                                        self.display_name(message.sender_id, message.chat_id) or "群友",
                                        tuple(reversed(previous)), message.text)

    @contextmanager
    def delivery_guard(self, snapshot):
        # 仅原生发送期间持锁，随机等待和大模型调用不会阻塞 /清空。
        with self._lock:
            if not self.snapshot_is_current(snapshot):
                raise SnapshotExpired("聊天记录已清空或已跨天")
            yield

    def _member(self, row):
        if row["person_key"] is None:
            return Member(row["room"], row["name"], row["name"], row["wxid"], (), False)
        aliases = self._connection().execute(
            "SELECT alias FROM aliases WHERE account=? AND room=? AND person_key=? ORDER BY alias",
            (self.account, row["room"], row["person_key"])).fetchall()
        return Member(row["room"], row["name"], row["display_name"], row["wxid"], tuple(r[0] for r in aliases),
                      bool(row["self_named"]))

    def remember_wechat_name(self, wxid, name):
        """微信原始名字按账号 ID 缓存，不产生任何群内绑定或别名。"""
        validate_account(wxid)
        normalized = name_key(name)
        if any(ord(ch) < 32 for ch in name):
            raise MemberError("微信名不能包含控制字符")
        with self._lock, self._connection() as db:
            db.execute("""INSERT INTO wechat_names VALUES(?,?,?,?) ON CONFLICT(account,wxid)
                DO UPDATE SET name=excluded.name,name_key=excluded.name_key""",
                (self.account, wxid, name.strip(), normalized))

    def wechat_name(self, wxid):
        with self._lock:
            row = self._connection().execute("SELECT name FROM wechat_names WHERE account=? AND wxid=?",
                                             (self.account, wxid)).fetchone()
            return row[0] if row else None

    def display_name(self, wxid, room):
        """群内绑定优先；其他群的绑定永远不作为默认名字。"""
        member = self.find_by_wxid(wxid, room)
        return member.display_name if member else self.group_nickname(wxid, room) or self.wechat_name(wxid)

    def remember_group_nickname(self, wxid, room, nickname):
        """微信群昵称按账号、群和成员隔离，不覆盖 /绑定 的称呼。"""
        validate_mentions(room, [wxid])
        if not isinstance(nickname, str) or len(nickname) > 64 or any(ord(ch) < 32 for ch in nickname):
            raise MemberError("群昵称应为最多 64 字符的单行文本")
        with self._lock, self._connection() as db:
            if nickname.strip():
                db.execute("""INSERT INTO group_nicknames VALUES(?,?,?,?)
                    ON CONFLICT(account,room,wxid) DO UPDATE SET nickname=excluded.nickname""",
                    (self.account, room, wxid, nickname.strip()))
            else:
                db.execute("DELETE FROM group_nicknames WHERE account=? AND room=? AND wxid=?",
                           (self.account, room, wxid))

    def group_nickname(self, wxid, room):
        with self._lock:
            row = self._connection().execute(
                "SELECT nickname FROM group_nicknames WHERE account=? AND room=? AND wxid=?",
                (self.account, room, wxid)).fetchone()
            return row[0] if row else None

    def _row(self, name, room=None):
        sql = """SELECT p.* FROM people p JOIN aliases a USING(account,room,person_key)
                 WHERE a.account=? AND a.alias=?"""
        values = [self.account, name_key(name)]
        if room is not None:
            sql += " AND a.room=?"
            values.append(room)
        rows = self._connection().execute(sql, values).fetchall()
        if not rows:
            # 未绑定也可按已核对的微信名查询，但必须确实在这个群说过话。
            query = """SELECT DISTINCT m.room,n.name,n.wxid FROM wechat_names n JOIN room_members m
                ON m.account=n.account AND m.wxid=n.wxid WHERE n.account=? AND n.name_key=?"""
            parameters = [self.account, name_key(name)]
            if room is not None:
                query += " AND m.room=?"
                parameters.append(room)
            names = self._connection().execute(query, parameters).fetchall()
            if len(names) > 1:
                raise MemberError(f"微信名 {name} 对应多个成员或群，请用本群绑定称呼查询")
            if not names:
                raise MemberError(f"未登记称呼：{name}")
            bound = self._connection().execute("SELECT * FROM people WHERE account=? AND room=? AND wxid=?",
                (self.account, names[0]["room"], names[0]["wxid"])).fetchone()
            if bound:
                return bound
            return {"room": names[0]["room"], "name": names[0]["name"], "wxid": names[0]["wxid"],
                    "person_key": None, "display_name": names[0]["name"], "self_named": 0}
        if len(rows) != 1:
            raise MemberError(f"多个群都有 {name}，请指定群 ID")
        return rows[0]

    def recent_room_lines(self, room, seconds, limit, *, clip=200):
        """最近几分钟的群文本，从旧到新。跳过指令和探测消息。clip 为 None 时不截断正文。"""
        from .summary import CONTROL
        from .text import own_group_command

        if not self.history_enabled or room not in self.groups:
            return []
        cutoff = (self._now() - timedelta(seconds=max(1, int(seconds)))).astimezone(timezone.utc).isoformat()
        with self._lock:
            rows = self._connection().execute(
                """SELECT id, sender_id, text, received_at, bot_generated, session_id, sequence
                   FROM messages WHERE account=? AND room=? AND received_at>=?
                   ORDER BY id DESC LIMIT ?""",
                (self.account, room, cutoff, max(1, int(limit)) * 3),
            ).fetchall()
            chosen = []
            for row in reversed(rows):
                text = " ".join((row["text"] or "").split())
                if not text or "WXBOT_AT_CHAT_PROBE" in text or own_group_command(text):
                    continue
                if text.split(maxsplit=1)[0] in CONTROL:
                    continue
                chosen.append({
                    "id": row["id"],
                    "sender_id": row["sender_id"],
                    "text": text if clip is None else text[:clip],
                    "bot": bool(row["bot_generated"]),
                    "session_id": row["session_id"] or "",
                    "sequence": row["sequence"],
                    "received_at": row["received_at"],
                    "speaker": self._context_speaker(room, row["sender_id"], row["bot_generated"]),
                    "age_seconds": self._line_age(row["received_at"]),
                })
        return chosen[-max(1, int(limit)):]

    def search_room_lines(self, room, keyword, days, limit=20):
        """按关键词查本群最近几天的聊天，从旧到新。跳过指令和探测消息。"""
        from .summary import CONTROL
        from .text import own_group_command

        keyword = " ".join(str(keyword or "").split())[:50]
        if not keyword or not self.history_enabled or room not in self.groups:
            return []
        cutoff = (self._now() - timedelta(days=min(max(int(days), 1), 30))).astimezone(timezone.utc).isoformat()
        pattern = "%" + keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._lock:
            rows = self._connection().execute(
                """SELECT sender_id, text, received_at, bot_generated FROM messages
                   WHERE account=? AND room=? AND received_at>=? AND text LIKE ? ESCAPE '\\'
                   ORDER BY id DESC LIMIT ?""",
                (self.account, room, cutoff, pattern, max(1, int(limit)) * 3),
            ).fetchall()
            chosen = []
            for row in rows:
                text = " ".join((row["text"] or "").split())
                if not text or "WXBOT_AT_CHAT_PROBE" in text or own_group_command(text):
                    continue
                if text.split(maxsplit=1)[0] in CONTROL:
                    continue
                chosen.append({"received_at": row["received_at"], "text": text[:200],
                               "speaker": self._context_speaker(room, row["sender_id"], row["bot_generated"])})
                if len(chosen) >= limit:
                    break
        chosen.reverse()
        return chosen

    def _line_age(self, received_at):
        try:
            moment = datetime.fromisoformat(received_at)
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return max(0, (self._now() - moment).total_seconds())

    def resolve(self, name, room=None):
        with self._lock:
            return self._member(self._row(name, room))

    def list_members(self, room=None):
        with self._lock:
            sql, values = "SELECT * FROM people WHERE account=?", [self.account]
            if room is not None:
                sql += " AND room=?"
                values.append(room)
            return [self._member(row) for row in self._connection().execute(sql + " ORDER BY room,person_key", values).fetchall()]

    def find_by_wxid(self, wxid, room):
        with self._lock:
            row = self._connection().execute("SELECT * FROM people WHERE account=? AND room=? AND wxid=?",
                                             (self.account, room, wxid)).fetchone()
            return self._member(row) if row else None

    def bind(self, name, room, wxid):
        validate_mentions(room, [wxid])
        with self._lock:
            row = self._row(name, room)
            if row["person_key"] is None:
                raise MemberError("请本人发送 /绑定 昵称 来设置本群称呼")
            if row["wxid"] and row["wxid"] != wxid:
                raise MemberError(f"{row['name']} 已绑定其他账号，未修改原绑定")
            other = self._connection().execute("SELECT name FROM people WHERE account=? AND room=? AND wxid=?",
                                                (self.account, room, wxid)).fetchone()
            if other and other["name"] != row["name"]:
                raise MemberError(f"这个微信账号已绑定为 {other['name']}，未修改原绑定")
            with self._connection() as db:
                db.execute("UPDATE people SET wxid=? WHERE account=? AND room=? AND person_key=?",
                           (wxid, self.account, room, row["person_key"]))
            return self._member(self._row(name, room))

    def register(self, name, room, wxid):
        """本人注册新称呼或改称呼；身份、旧称呼和记录仍由同一个 ID 关联。"""
        alias = name_key(name)
        name = name.strip()
        validate_mentions(room, [wxid])
        if room not in self.groups:
            raise MemberError("这个群尚未启用成员目录")
        with self._lock:
            db = self._connection()
            by_name = db.execute("""SELECT p.* FROM people p JOIN aliases a USING(account,room,person_key)
                WHERE a.account=? AND a.room=? AND a.alias=?""", (self.account, room, alias)).fetchone()
            by_id = db.execute("SELECT * FROM people WHERE account=? AND room=? AND wxid=?",
                               (self.account, room, wxid)).fetchone()
            if by_name and by_name["wxid"] and by_name["wxid"] != wxid:
                raise MemberError(f"称呼 {name} 已被其他成员绑定，请换一个昵称")
            if by_name and by_id and by_name["person_key"] != by_id["person_key"]:
                raise MemberError(f"称呼 {name} 已属于另一份成员资料，请换一个昵称")
            row = by_id or by_name
            key = row["person_key"] if row else "member_" + uuid.uuid4().hex
            display = name  # 本人 /绑定 的名字同时用于本群查询和艾特显示。
            with db:
                if row:
                    db.execute("UPDATE people SET name=?,display_name=?,wxid=?,self_named=1 WHERE account=? AND room=? AND person_key=?",
                               (name, display, wxid, self.account, room, key))
                else:
                    db.execute("""INSERT INTO people(account,room,person_key,name,display_name,wxid,self_named)
                        VALUES(?,?,?,?,?,?,1)""", (self.account, room, key, name, name, wxid))
                db.execute("INSERT OR IGNORE INTO aliases VALUES(?,?,?,?)", (self.account, room, alias, key))
            return self._member(self._row(name, room))

    def record_bot_reply(self, message, text):
        """把机器人刚发出的回复写进群流水，下次上下文才能看见自己说过的话。"""
        if not isinstance(text, str) or not text.strip():
            return False
        reply = type(message)(text.strip(), message.chat_id, sender_id=self.account, is_group=True,
                              is_self=True, session_id=message.session_id, sequence=0,
                              data={"bot_generated": True})
        return self.record(reply)

    def record(self, message):
        # 引用回复的新正文也是群聊文字，一并记下，上下文才连得上。
        quote = message.type == 49 and message.data.get("app_type") == 57 and bool(str(message.text).strip())
        if not self.history_enabled or (message.type != 1 and not quote) or not message.is_group or message.chat_id not in self.groups:
            return False
        if message.data.get("self_wxid", self.account) != self.account:
            return False
        validate_mentions(message.chat_id, [message.sender_id])
        if not isinstance(message.text, str) or not message.session_id:
            raise MemberError("群记录缺少正文或连接标识")
        native_name = message.data.get("sender_wechat_name")
        if native_name:
            self.remember_wechat_name(message.sender_id, native_name)
        with self._lock, self._connection() as db:
            self.rollover()
            db.execute("INSERT OR IGNORE INTO room_members VALUES(?,?,?)",
                       (self.account, message.chat_id, message.sender_id))
            result = db.execute("""INSERT OR IGNORE INTO messages
                (account,room,sender_id,text,received_at,session_id,sequence,is_self,bot_generated) VALUES(?,?,?,?,?,?,?,?,?)""",
                (self.account, message.chat_id, message.sender_id, message.text,
                 self._now().astimezone(timezone.utc).isoformat(), message.session_id, message.sequence,
                 int(message.is_self), int(bool(message.data.get("bot_generated")))))
            return result.rowcount > 0

    def history(self, name, room=None, *, limit=100, since=None):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise MemberError("历史查询 limit 应为 1–1000 的整数")
        if since is not None:
            if not isinstance(since, datetime) or since.tzinfo is None or since.utcoffset() is None:
                raise MemberError("since 应为带时区的 datetime")
            since = since.astimezone(timezone.utc).isoformat()
        with self._lock:
            self.rollover()
            member = self.resolve(name, room)
            if member.wxid is None:
                raise MemberError(f"{member.name} 尚未绑定微信 ID")
            sql = "SELECT * FROM messages WHERE account=? AND room=? AND sender_id=?"
            values = [self.account, member.room, member.wxid]
            if since is not None:
                sql += " AND received_at>=?"
                values.append(since)
            rows = self._connection().execute(sql + " ORDER BY id DESC LIMIT ?", [*values, limit]).fetchall()
            return [dict(row) for row in reversed(rows)]

    def close(self):
        with self._lock:
            if self._db:
                self._db.close()
                self._db = None
            self._closed = True
