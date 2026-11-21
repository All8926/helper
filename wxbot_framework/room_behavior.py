"""群管理员、点名别名和人格提示词的校验与匹配。"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from string import Formatter

from .account import validate_account

RESERVED_PERSONALITY = "默认"
MAX_ADMINS = 32
MAX_ALIASES = 8
MAX_PERSONALITIES = 128
MAX_LAYER = 2000
MAX_VOICE_LIBRARY = 24
DEFAULT_CONTEXT_HOURS = 6
DEFAULT_CONTEXT_LIMIT = 300
MAX_CONTEXT_HOURS = 168
MAX_CONTEXT_LIMIT = 2000
DEFAULT_GLANCE_MINUTES = 2
MAX_GLANCE_MINUTES = 60
DEFAULT_HOURLY_LIMIT = 3
MAX_HOURLY_LIMIT = 30
ALIAS_TEXT = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9_-]{2,16}$")
PERSONALITY_NAME = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9_-]{1,16}$")
DEFAULT_REPLY_LIMIT = 600
MAX_REPLY_LIMIT = 8000
# 所有私聊共用这一行人格代际，不能当成群 ID。
PRIVATE_PERSONALITY_ROOM = "__private__"
MAX_PRIVATE_CONTACTS = 1000


class BehaviorError(ValueError):
    pass


def normalize_personalities(raw):
    """整理人格库。没有配置就是空的，不预置任何人格。旧的整段提示词直接丢掉。"""
    from .personality import LAYER_NAMES
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_PERSONALITIES:
        raise BehaviorError(f"人格最多 {MAX_PERSONALITIES} 份")
    result, seen = [], set()
    for item in raw:
        if not isinstance(item, dict) or "layers" not in item:
            continue
        name = item.get("name")
        if not isinstance(name, str) or not PERSONALITY_NAME.fullmatch(name.strip()):
            raise BehaviorError("人格名称应为 1–16 个汉字、字母或数字")
        name = name.strip()
        if name == RESERVED_PERSONALITY:
            raise BehaviorError("「默认」表示不使用人格，不能作为人格名称")
        raw_layers = item.get("layers")
        if not isinstance(raw_layers, dict):
            raise BehaviorError("人格层面应为文本")
        if set(raw_layers) - set(LAYER_NAMES):
            raise BehaviorError("人格层面名称无效")
        layers = {}
        for key in LAYER_NAMES:
            value = raw_layers.get(key, "")
            if value is None:
                value = ""
            if not isinstance(value, str) or "\0" in value:
                raise BehaviorError(f"{name}的{key}应为文本")
            text = value.strip()
            if len(text) > MAX_LAYER:
                raise BehaviorError(f"{name}的{key}不超过 {MAX_LAYER} 字")
            layers[key] = text
        voice = item.get("voice", "")
        if voice is None:
            voice = ""
        if not isinstance(voice, str) or (voice.strip() and not PERSONALITY_NAME.fullmatch(voice.strip())):
            raise BehaviorError("人格音色应留空，或选择音色库里的名称")
        if name in seen:
            raise BehaviorError(f"人格名称重复：{name}")
        previous = item.get("previous", name)
        if previous is None:
            previous = ""
        if not isinstance(previous, str) or len(previous) > 16:
            raise BehaviorError("原人格名称无效")
        seen.add(name)
        result.append({"name": name, "layers": layers, "voice": voice.strip(), "previous": previous.strip()})
    return result


def public_personalities(items):
    return [{"name": item["name"], "layers": dict(item["layers"]), "voice": item.get("voice", "")} for item in items]


def normalize_voice_library(raw):
    """整理音色库。没配置时为空，不预置音色。"""
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_VOICE_LIBRARY:
        raise BehaviorError("音色最多 24 个")
    from .tts import VOICE_ID, VOICE_ID_HINT, normalize_voice_id
    result, seen = [], set()
    reserved = {"关闭", "文字", "自动", "始终"}
    for item in raw:
        if not isinstance(item, dict):
            raise BehaviorError("音色应为名称和音色 ID")
        name = item.get("name")
        if not isinstance(name, str) or not PERSONALITY_NAME.fullmatch(name.strip()):
            raise BehaviorError("音色名称应为 1–16 个汉字、字母或数字")
        name = name.strip()
        if name in reserved:
            raise BehaviorError(f"「{name}」不能作为音色名称")
        if not isinstance(item.get("voice_id"), str):
            raise BehaviorError(VOICE_ID_HINT)
        voice_id = normalize_voice_id(item.get("voice_id"))
        if not VOICE_ID.fullmatch(voice_id):
            raise BehaviorError(VOICE_ID_HINT)
        if name in seen:
            raise BehaviorError(f"音色名称重复：{name}")
        seen.add(name)
        result.append({"name": name, "voice_id": voice_id})
    return result


def normalize_global_prompt(value):
    """对话任务说明。没写或留空就是空，不填回内置句子，也不限制字数。"""
    if value is None:
        return ""
    if not isinstance(value, str) or "\0" in value:
        raise BehaviorError("全局提示词应为不含空字符的文本")
    return value.strip()


def normalize_reply_limit(value):
    """艾特回复和聊天总结的字数上限。没填时是 600。"""
    if value is None:
        return DEFAULT_REPLY_LIMIT
    if type(value) is not int or isinstance(value, bool) or not 1 <= value <= MAX_REPLY_LIMIT:
        raise BehaviorError("回复字数上限应为 1–8000 的整数")
    return value


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


def normalize_unnamed_reply(value, setting=None):
    """未点名回应开关。没写过时看旧配置：活跃度大于 0 或打开过主动插话的群视为开启。"""
    if value is None:
        legacy = setting if isinstance(setting, dict) else {}
        activity = legacy.get("activity")
        return (type(activity) is int and activity > 0) or legacy.get("proactive_enabled") is True
    if type(value) is not bool:
        raise BehaviorError("未点名回应应为开或关")
    return value


def normalize_hourly_limit(value):
    """每小时按节拍主动说话最多几句。没填时 3 句。"""
    if value is None:
        return DEFAULT_HOURLY_LIMIT
    if type(value) is not int or not 0 <= value <= MAX_HOURLY_LIMIT:
        raise BehaviorError("每小时上限应为 0–30 的整数")
    return value


def normalize_deep_thinking(value):
    """没填时开启。只对 DeepSeek 官方接口生效。"""
    if value is None:
        return True
    if type(value) is not bool:
        raise BehaviorError("深度思考应为开或关")
    return value


DEFAULT_SPEAK_THRESHOLD = 0.5
DEFAULT_VOICE_THRESHOLD = 0.6
DEFAULT_SING_THRESHOLD = 0.85


def normalize_speak_threshold(value):
    """未点名回应时，决策概率达到这个数才说话。没填时 0.5。0 表示模型给出的概率都算该说，1 表示只有 1 才说。"""
    if value is None:
        return DEFAULT_SPEAK_THRESHOLD
    if type(value) is bool or type(value) not in (int, float) or value != value or not 0 <= value <= 1:
        raise BehaviorError("说话门槛应为 0–1 之间的数")
    return round(float(value), 2)


def normalize_voice_threshold(value, setting=None):
    """自动语音时，决策概率达到这个数才念出来。没填时 0.6。旧的语音频率 1–10 按原换算保留。"""
    if value is None:
        legacy = setting.get("voice_rate") if isinstance(setting, dict) else None
        if type(legacy) is int and 1 <= legacy <= 10:
            return round(0.85 - 0.05 * legacy, 2)
        return DEFAULT_VOICE_THRESHOLD
    if type(value) is bool or type(value) not in (int, float) or value != value or not 0 <= value <= 1:
        raise BehaviorError("语音门槛应为 0–1 之间的数")
    return round(float(value), 2)


def normalize_sing_threshold(value):
    """清唱时，决策概率达到这个数才把正文当成歌词。没填时 0.85。"""
    if value is None:
        return DEFAULT_SING_THRESHOLD
    if type(value) is bool or type(value) not in (int, float) or value != value or not 0 <= value <= 1:
        raise BehaviorError("清唱门槛应为 0–1 之间的数")
    return round(float(value), 2)


def normalize_glance_minutes(value):
    """未点名回应隔几分钟看一眼群。没填时 2 分钟，0 表示不按节拍看，只在有人说完并安静下来时判断。"""
    if value is None:
        return DEFAULT_GLANCE_MINUTES
    if type(value) is not int or not 0 <= value <= MAX_GLANCE_MINUTES:
        raise BehaviorError("看群间隔应为 0–60 的整数分钟")
    return value


def normalize_notice_template(value, field):
    """进退群文案只允许 {nickname}，空文本表示继续用账号里的模板。"""
    label = "进群文案" if field == "welcome_template" else "退群文案"
    if not isinstance(value, str):
        raise BehaviorError(f"{label}应为文本")
    text = value.strip()
    if not text:
        return ""
    if len(text.encode("utf-8")) > 700 or "\0" in text:
        raise BehaviorError(f"{label}不能为空，且不超过 700 个字节")
    try:
        for _, name, spec, conversion in Formatter().parse(text):
            if name is not None and (name != "nickname" or spec or conversion):
                raise BehaviorError(f"{label}只支持 {{nickname}}")
    except BehaviorError:
        raise
    except ValueError:
        raise BehaviorError(f"{label}格式不正确") from None
    return text


def normalize_plugin_options(value):
    """只保存这个群相对账号配置改过的项。目前开放进退群提醒的开关和文案。"""
    if value is None:
        value = {}
    if not isinstance(value, dict) or len(value) > 8:
        raise BehaviorError("本群插件选项过多")
    from .config import validate_name
    result = {}
    for name, options in value.items():
        try:
            plugin = validate_name(name)
        except ValueError as error:
            raise BehaviorError("插件名称无效") from error
        if plugin != "group_notice":
            raise BehaviorError("目前只能为进退群提醒按群设置文案")
        cleaned = _normalize_notice_options(options)
        if cleaned:
            result[plugin] = cleaned
    return result


def _normalize_notice_options(options):
    allowed = {"welcome", "leave_notice", "welcome_template", "leave_template"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise BehaviorError("进退群文案格式不正确")
    cleaned = {}
    for key, label in (("welcome", "进群欢迎"), ("leave_notice", "退群提醒")):
        if key not in options:
            continue
        if type(options[key]) is not bool:
            raise BehaviorError(f"{label}应为开或关")
        cleaned[key] = options[key]
    for key in ("welcome_template", "leave_template"):
        if key not in options:
            continue
        text = normalize_notice_template(options[key], key)
        if text:
            cleaned[key] = text
    return cleaned


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
            "context_limit": normalize_context_limit(value.get("context_limit")),
            "unnamed_reply": normalize_unnamed_reply(value.get("unnamed_reply"), value),
            "glance_minutes": normalize_glance_minutes(value.get("glance_minutes")),
            "hourly_limit": normalize_hourly_limit(value.get("hourly_limit")),
            "speak_threshold": normalize_speak_threshold(value.get("speak_threshold")),
            "voice_threshold": normalize_voice_threshold(value.get("voice_threshold"), value),
            "sing_threshold": normalize_sing_threshold(value.get("sing_threshold")),
            "deep_thinking": normalize_deep_thinking(value.get("deep_thinking")),
            "plugin_options": normalize_plugin_options(value.get("plugin_options"))}


def _private_personality_name(value, personality_names, *, fallback):
    if not isinstance(value, str):
        raise BehaviorError("请选择已保存的人格")
    personality = value.strip()
    if personality != RESERVED_PERSONALITY and personality not in personality_names:
        if not fallback:
            raise BehaviorError("请选择已保存的人格")
        return RESERVED_PERSONALITY
    return personality


def normalize_private_person(value, personality_names, *, fallback=False, legacy_name="", legacy_personality=None):
    """一个人的名称、人格、深度思考和语音。名称可以留空。"""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise BehaviorError("联系人配置格式不正确")
    name = value.get("name", legacy_name)
    if name is None:
        name = ""
    if not isinstance(name, str) or len(name) > 64 or any(ord(ch) < 32 for ch in name):
        raise BehaviorError("联系人名称不超过 64 个字符")
    personality = value.get("personality", legacy_personality if legacy_personality is not None else RESERVED_PERSONALITY)
    mode = value.get("voice_mode", "off")
    if mode not in ("off", "always", "auto"):
        raise BehaviorError("语音回复应为关闭、始终语音或自动")
    voice_id = value.get("voice_id", "")
    if voice_id is None:
        voice_id = ""
    if not isinstance(voice_id, str):
        raise BehaviorError("音色 ID 无效")
    from .tts import VOICE_ID, VOICE_ID_HINT, normalize_voice_id
    voice_id = normalize_voice_id(voice_id)
    if voice_id and not VOICE_ID.fullmatch(voice_id):
        raise BehaviorError(VOICE_ID_HINT)
    return {"name": name.strip(),
            "personality": _private_personality_name(personality, personality_names, fallback=fallback),
            "deep_thinking": normalize_deep_thinking(value.get("deep_thinking")),
            "voice_mode": mode,
            "voice_id": voice_id,
            "voice_threshold": normalize_voice_threshold(value.get("voice_threshold"), value),
            "sing_threshold": normalize_sing_threshold(value.get("sing_threshold"))}


def normalize_private_chat(value, personality_names, *, fallback=False):
    """私聊名单、共用上下文，以及每个人自己的名称、人格和语音。"""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise BehaviorError("私聊配置格式不正确")
    raw_contacts = value.get("contacts", [])
    if not isinstance(raw_contacts, list) or len(raw_contacts) > MAX_PRIVATE_CONTACTS:
        raise BehaviorError("私聊联系人应为列表，最多 1000 个")
    contacts, seen = [], set()
    for item in raw_contacts:
        if not isinstance(item, str):
            raise BehaviorError("私聊联系人应为微信 ID")
        try:
            wxid = validate_account(item.strip())
        except ValueError as error:
            raise BehaviorError("私聊联系人必须是个人微信 ID") from error
        if wxid not in seen:
            seen.add(wxid)
            contacts.append(wxid)
    raw_names = value.get("names") or {}
    if not isinstance(raw_names, dict):
        raise BehaviorError("联系人名称格式不正确")
    shared = value.get("personality", RESERVED_PERSONALITY)
    if not isinstance(shared, str):
        shared = RESERVED_PERSONALITY
    raw_people = value.get("people") or {}
    if not isinstance(raw_people, dict):
        raise BehaviorError("联系人配置格式不正确")
    people = {}
    for wxid in contacts:
        raw = raw_people.get(wxid) if isinstance(raw_people.get(wxid), dict) else None
        legacy_name = raw_names.get(wxid, "") if isinstance(raw_names.get(wxid), str) else ""
        people[wxid] = normalize_private_person(
            raw, personality_names, fallback=fallback, legacy_name=legacy_name,
            legacy_personality=shared if raw is None or "personality" not in raw else None)
    names = {wxid: person["name"] for wxid, person in people.items() if person["name"]}
    return {"contacts": contacts, "names": names, "people": people,
            "context_hours": normalize_context_hours(value.get("context_hours")),
            "context_limit": normalize_context_limit(value.get("context_limit"))}


def view_behavior(options):
    """给管理台的人格库和各群行为。没配人格就是空库。"""
    if not isinstance(options, dict):
        options = {}
    try:
        items = normalize_personalities(options["personalities"] if "personalities" in options else None)
    except (BehaviorError, TypeError, KeyError):
        items = []
    settings = options.get("room_settings")
    try:
        guard = normalize_global_prompt(options["global_prompt"] if "global_prompt" in options else None)
    except (BehaviorError, TypeError):
        guard = ""
    try:
        limit = normalize_reply_limit(options.get("reply_limit"))
    except (BehaviorError, TypeError):
        limit = DEFAULT_REPLY_LIMIT
    known = [item["name"] for item in items]
    try:
        private_chat = normalize_private_chat(options.get("private_chat"), known, fallback=True)
    except (BehaviorError, TypeError, ValueError):
        private_chat = normalize_private_chat(None, known)
    try:
        voices = normalize_voice_library(options["voice_library"] if "voice_library" in options else None)
    except (BehaviorError, TypeError):
        voices = []
    return {"personalities": public_personalities(items),
            "global_prompt": guard,
            "reply_limit": limit,
            "voice_library": voices,
            "room_settings": settings if isinstance(settings, dict) else {},
            "private_chat": private_chat}


def command_allowed(admins, sender, robot):
    """还没指定管理员时不限制。机器人自己的微信 ID 始终可以发指令。"""
    if not admins:
        return True
    return sender == robot or sender in admins


def is_slash_command(text):
    return isinstance(text, str) and text.strip().startswith("/")


def chat_switch(text):
    """只认整句指令。True 是开启本群艾特对话，False 是关闭。"""
    if not isinstance(text, str):
        return None
    command = text.strip()
    if command == "/开启对话":
        return True
    if command == "/关闭对话":
        return False
    return None


_ACTIVITY_NUMBER = re.compile(r"^(?:0(?:\.\d+)?|1(?:\.0+)?)$")


def speak_activity(text):
    """解析 /对话活跃度。不是这条指令时返回 None，数值不在 0–1 时返回 False。"""
    if not isinstance(text, str):
        return None
    parts = text.strip().split()
    if not parts or parts[0] != "/对话活跃度":
        return None
    if len(parts) != 2 or _ACTIVITY_NUMBER.fullmatch(parts[1]) is None:
        return False
    value = float(parts[1])
    if value != value or not 0 <= value <= 1:
        return False
    return round(value, 2)


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
