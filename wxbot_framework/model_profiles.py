"""大模型和语音各自的配置库。正在使用的那份仍写回原来的配置文件。"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

FILENAME = "model_profiles.json"
MAX_PROFILES = 12
SYNTHETIC = {"llm": "current-llm", "tts": "current-tts"}
FIELDS = {
    "llm": ("base_url", "model", "timeout_seconds", "max_tokens", "json_mode",
            "token_limit_parameter", "extra_body"),
    "tts": ("base_url", "model", "timeout_seconds", "default_voice", "max_chars_per_clip"),
}


class ProfileError(ValueError):
    pass


def blank():
    return {"llm": [], "tts": [], "active": {"llm": "", "tts": ""}}


def load_profiles(path):
    path = Path(path)
    if not path.is_file():
        return blank()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return blank()
    if not isinstance(raw, dict):
        return blank()
    library = blank()
    for kind in ("llm", "tts"):
        items = raw.get(kind) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and isinstance(item.get("name"), str):
                library[kind].append(item)
        chosen = raw.get("active", {}).get(kind) if isinstance(raw.get("active"), dict) else ""
        ids = [item["id"] for item in library[kind]]
        library["active"][kind] = chosen if chosen in ids else (ids[0] if ids else "")
    return library


def view_library(library, documents):
    """给页面的配置列表。密钥只保留是否已配置，不返回原文。"""
    shown = {"active": {}}
    for kind in ("llm", "tts"):
        items = list(library.get(kind) or [])
        active = (library.get("active") or {}).get(kind) or ""
        if not items:
            items = [_stored(kind, documents.get(kind) or {}, "当前配置", SYNTHETIC[kind])]
            active = SYNTHETIC[kind]
        elif active not in {item["id"] for item in items}:
            active = items[0]["id"]
        shown[kind] = [_public(kind, item) for item in items]
        shown["active"][kind] = active
    return shown


def apply_profile(library, kind, request, incoming, current_file):
    """保存、新增、切换或删除一份配置。返回配置库，以及需要写回当前文件的内容。"""
    if kind not in FIELDS:
        raise ProfileError("配置类型不正确")
    items = list(library.get(kind) or [])
    active_id = (library.get("active") or {}).get(kind) or ""
    if request is None:
        merged = _merge(kind, current_file, incoming)
        _validate(kind, merged)
        if not items:
            item = _stored(kind, merged, "当前配置", SYNTHETIC[kind])
            library[kind] = [item]
            library["active"][kind] = item["id"]
        else:
            current = _find(items, active_id) or items[0]
            updated = _stored(kind, merged, current["name"], current["id"])
            library[kind] = [updated if item["id"] == current["id"] else item for item in items]
            library["active"][kind] = current["id"]
        return library, _overlay(current_file, merged, kind)
    if not isinstance(request, dict):
        raise ProfileError("配置切换格式不正确")
    action = request.get("action") or "save"
    if action not in ("save", "use", "add", "delete"):
        raise ProfileError("不支持的配置操作")
    if not items:
        seed = _stored(kind, current_file, "当前配置", SYNTHETIC[kind])
        items = [seed]
        library[kind] = items
        active_id = seed["id"]
        library["active"][kind] = active_id
    if action == "delete":
        target = request.get("id")
        if len(items) < 2:
            raise ProfileError("至少保留一份配置")
        if _find(items, target) is None:
            raise ProfileError("找不到要删除的配置")
        library[kind] = [item for item in items if item["id"] != target]
        if active_id == target:
            nxt = library[kind][0]
            library["active"][kind] = nxt["id"]
            return library, _overlay(current_file, nxt, kind)
        return library, None
    if action == "add":
        if len(items) >= MAX_PROFILES:
            raise ProfileError("每种配置最多 12 份")
        source = _find(items, request.get("id")) or _find(items, active_id) or items[0]
        merged = _merge(kind, source, incoming or {})
        _validate(kind, merged)
        item = _stored(kind, merged, _name(request.get("name") or "新配置", items, suffix=True), uuid.uuid4().hex[:12])
        library[kind] = [*items, item]
        return library, None
    current = _find(items, request.get("id") or active_id)
    if current is None:
        raise ProfileError("找不到要保存的配置")
    merged = _merge(kind, current, incoming or {})
    _validate(kind, merged)
    name = _name(request.get("name") or current["name"], items, own_id=current["id"])
    updated = _stored(kind, merged, name, current["id"])
    library[kind] = [updated if item["id"] == current["id"] else item for item in items]
    if action == "use" or current["id"] == library["active"][kind]:
        library["active"][kind] = current["id"]
        return library, _overlay(current_file, merged, kind)
    return library, None


def _public(kind, item):
    view = {key: item[key] for key in FIELDS[kind] if key in item}
    view.update(id=item["id"], name=item["name"], api_key_configured=bool(item.get("api_key")))
    if kind == "llm":
        view["api_key"] = item.get("api_key") or ""
    return view


def _stored(kind, doc, name, identifier):
    item = {"id": identifier, "name": name, "api_key": doc.get("api_key") or ""}
    for key in FIELDS[kind]:
        if key in doc:
            item[key] = doc[key]
    return item


def _find(items, identifier):
    return next((item for item in items if item["id"] == identifier), None)


def _merge(kind, base, incoming):
    doc = {key: base[key] for key in FIELDS[kind] if key in base}
    doc["api_key"] = base.get("api_key") or ""
    if not incoming:
        return doc
    for key in FIELDS[kind]:
        if key in incoming:
            doc[key] = incoming[key]
    key = incoming.get("api_key", "")
    if not isinstance(key, str) or len(key) > 1024 or any(ord(char) < 32 for char in key):
        raise ProfileError("API 密钥格式不正确")
    if incoming.get("clear_api_key") is True:
        doc["api_key"] = ""
    elif key.strip():
        doc["api_key"] = key.strip()
    if kind == "llm" and "base_url" in incoming:
        from .llm import is_worldguide_url
        # 跨到 WorldGuide 或离开它时，不能沿用另一家的密钥。
        if (doc.get("api_key") and not key.strip() and incoming.get("clear_api_key") is not True
                and is_worldguide_url(base.get("base_url")) != is_worldguide_url(doc.get("base_url"))):
            raise ProfileError("切换 WorldGuide 接口时，请填写新接口的 API 密钥或清除已保存的密钥")
    return doc


def _overlay(existing, doc, kind):
    merged = dict(existing)
    for key in FIELDS[kind]:
        if key in doc:
            merged[key] = doc[key]
    merged["api_key"] = doc.get("api_key") or ""
    return merged


def _name(value, items, *, own_id=None, suffix=False):
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        text = "新配置"
    if len(text) > 32 or any(ord(char) < 32 for char in text):
        raise ProfileError("配置名称应为 1–32 个字符")
    taken = {item["name"] for item in items if item["id"] != own_id}
    if text not in taken:
        return text
    if not suffix:
        raise ProfileError("已有同名配置")
    for index in range(2, 20):
        candidate = f"{text} {index}"
        if candidate not in taken:
            return candidate
    raise ProfileError("同名配置太多")


def _validate(kind, doc):
    trial = dict(doc, api_key=doc.get("api_key") or "validation-only")
    if kind == "llm":
        from .llm import LLMClient
        LLMClient(trial)
    else:
        from .tts import FishTTS
        FishTTS.from_options(trial)
