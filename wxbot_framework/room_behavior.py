"""群管理员、点名别名和人格提示词的校验与匹配。"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .account import validate_account

RESERVED_PERSONALITY = "默认"
MAX_ADMINS = 32
MAX_ALIASES = 8
MAX_PERSONALITIES = 12
MAX_PROMPT = 8000
DEFAULT_CONTEXT_HOURS = 6
DEFAULT_CONTEXT_LIMIT = 300
MAX_CONTEXT_HOURS = 168
MAX_CONTEXT_LIMIT = 2000
ALIAS_TEXT = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9_-]{2,16}$")
PERSONALITY_NAME = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9_-]{1,16}$")


class BehaviorError(ValueError):
    pass


def default_personalities():
    from .personality import PROMPTS
    return [{"name": name, "prompt": text, "previous": name} for name, text in PROMPTS.items()]


def normalize_personalities(raw):
    """整理人格库。未配置时使用内置的嘴臭、卑微、夸夸。"""
    if raw is None:
        return default_personalities()
    if not isinstance(raw, list) or len(raw) > MAX_PERSONALITIES:
        raise BehaviorError("人格最多 12 份")
    result, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise BehaviorError("人格应为名称和提示词")
        name = item.get("name")
        prompt = item.get("prompt")
        if not isinstance(name, str) or not PERSONALITY_NAME.fullmatch(name.strip()):
            raise BehaviorError("人格名称应为 1–16 个汉字、字母或数字")
        name = name.strip()
        if name == RESERVED_PERSONALITY:
            raise BehaviorError("「默认」表示不使用人格，不能作为人格名称")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > MAX_PROMPT:
            raise BehaviorError("人格提示词不能为空，且不超过 8000 字")
        if name in seen:
            raise BehaviorError(f"人格名称重复：{name}")
        previous = item.get("previous", name)
        if previous is None:
            previous = ""
        if not isinstance(previous, str) or len(previous) > 16:
            raise BehaviorError("原人格名称无效")
        seen.add(name)
        result.append({"name": name, "prompt": prompt.strip(), "previous": previous.strip()})
    return result


def public_personalities(items):
    return [{"name": item["name"], "prompt": item["prompt"]} for item in items]


def normalize_admins(values):
    if not isinstance(values, list) or len(values) > MAX_ADMINS:
        raise BehaviorError("管理员应为微信 ID 列表，最多 32 个")
    result = []
    for value in values:
        if not isinstance(value, str):
            raise BehaviorError("管理员应为微信 ID")
        result.append(validate_account(value.strip()))
    if len(set(result)) != len(result):
        raise BehaviorError("管理员微信 ID 不能重复")
    return result


def normalize_aliases(values):
    if not isinstance(values, list) or len(values) > MAX_ALIASES:
        raise BehaviorError("别名最多 8 个")
    result, seen = [], set()
    for value in values:
        if not isinstance(value, str) or not ALIAS_TEXT.fullmatch(value.strip()):
            raise BehaviorError("别名应为 2–16 个汉字、字母或数字，不能有空格")
        alias = value.strip()
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(alias)
    return result


def normalize_disabled_plugins(values):
    """只记录这个群要关掉的插件。没写的插件继续跟随账号总开关。"""
    from .config import validate_name
    if values is None:
        values = []
    if not isinstance(values, list) or len(values) > 64:
        raise BehaviorError("每个群最多单独关闭 64 个插件")
    result = []
    for value in values:
        try:
            name = validate_name(value)
        except ValueError as error:
            raise BehaviorError("插件名称无效") from error
        if name not in result:
            result.append(name)
    return result


def normalize_context_hours(value):
    """没填时用 6 小时。每个群可以单独改。"""
    if value is None:
        return DEFAULT_CONTEXT_HOURS
    if type(value) is not int or not 1 <= value <= MAX_CONTEXT_HOURS:
        raise BehaviorError("上下文小时数应为 1–168 的整数")
    return value


def normalize_context_limit(value):
    """没填时最多带 300 条。每个群可以单独改。"""
    if value is None:
        return DEFAULT_CONTEXT_LIMIT
    if type(value) is not int or not 1 <= value <= MAX_CONTEXT_LIMIT:
        raise BehaviorError("上下文条数应为 1–2000 的整数")
    return value


def normalize_room_setting(value, names):
    if not isinstance(value, dict):
        raise BehaviorError("群行为配置格式不正确")
    personality = value.get("personality", RESERVED_PERSONALITY)
    if not isinstance(personality, str) or (personality != RESERVED_PERSONALITY and personality not in names):
        raise BehaviorError("请选择已保存的人格")
    return {"admins": normalize_admins(value.get("admins", [])),
            "aliases": normalize_aliases(value.get("aliases", [])),
            "personality": personality,
            "disabled_plugins": normalize_disabled_plugins(value.get("disabled_plugins", [])),
            "context_hours": normalize_context_hours(value.get("context_hours")),
            "context_limit": normalize_context_limit(value.get("context_limit"))}


def view_behavior(options):
    """给管理台的人格库和各群行为。配置缺失时补上内置人格。"""
    if not isinstance(options, dict):
        options = {}
    try:
        items = normalize_personalities(options["personalities"] if "personalities" in options else None)
    except (BehaviorError, TypeError, KeyError):
        items = default_personalities()
    settings = options.get("room_settings")
    return {"personalities": public_personalities(items),
            "room_settings": settings if isinstance(settings, dict) else {}}


def command_allowed(admins, sender, robot):
    """还没指定管理员时不限制。机器人自己的微信 ID 始终可以发指令。"""
    if not admins:
        return True
    return sender == robot or sender in admins


def is_slash_command(text):
    return isinstance(text, str) and text.strip().startswith("/")


def command_blocked(store, message):
    if (store is None or message.type != 1 or not message.is_group or
            message.chat_id not in store.groups or message.data.get("bot_generated") or
            not is_slash_command(message.text)):
        return False
    return not command_allowed(store.room_admins(message.chat_id), message.sender_id, store.account)


def alias_mentioned(text, aliases):
    """手写别名出现在正文里即算点名。英文按词边界匹配，避免嵌在更长的单词里。"""
    if not isinstance(text, str) or not text.strip() or not aliases:
        return False
    folded = text.casefold()
    for alias in aliases:
        if alias.isascii():
            if _ascii_contains(folded, alias.casefold()):
                return True
        elif alias in text:
            return True
    return False


def _ascii_contains(content, name):
    start = 0
    while True:
        index = content.find(name, start)
        if index < 0:
            return False
        before = index == 0 or not content[index - 1].isalnum()
        end = index + len(name)
        after = end >= len(content) or not content[end].isalnum()
        if before and after:
            return True
        start = index + 1


def write_json(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".room-behavior-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
