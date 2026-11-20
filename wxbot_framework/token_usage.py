"""按机器人记下对话接口返回的 token，并执行每日上限。

记录放在工作区 .runtime 里，和模型配置分开。切换模型配置不会清掉用量。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading

# 与成员目录同一天界，按北京时间切日。
BEIJING = timezone(timedelta(hours=8))
KEEP_DAYS = 30
MAX_DAILY_LIMIT = 100_000_000
_MAX_REPORTED = 1_000_000_000
_MAX_STORED = 10 ** 15

_books = {}
_books_guard = threading.Lock()


def open_usage(path):
    """同一路径共用一把锁，避免页面保存和模型入账互相覆盖。"""
    path = Path(path).resolve()
    with _books_guard:
        book = _books.get(path)
        if book is None:
            book = TokenUsage(path)
            _books[path] = book
        return book


def _count(value):
    if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value):
        return None
    if value < 0 or value > _MAX_REPORTED or (type(value) is float and not value.is_integer()):
        return None
    return int(value)


def usage_from(result):
    """取出服务商返回的用量。没有 usage 时返回 None。"""
    usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt = _count(usage.get("prompt_tokens"))
    completion = _count(usage.get("completion_tokens"))
    total = _count(usage.get("total_tokens"))
    if prompt is None and completion is None and total is None:
        return None
    prompt = 0 if prompt is None else prompt
    completion = 0 if completion is None else completion
    if total is None:
        total = prompt + completion
    return prompt, completion, total


def _bounded(value):
    if type(value) is int and 0 <= value <= MAX_DAILY_LIMIT:
        return value
    return 0


def _blank():
    return {"prompt": 0, "completion": 0, "total": 0, "calls": 0, "unreported": 0}


def _parse_day(value):
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        day = datetime.fromisoformat(value).date()
    except ValueError:
        return None
    if day.isoformat() != value:
        return None
    return day


def _row(raw):
    if not isinstance(raw, dict):
        return None
    row = _blank()
    for key in row:
        value = raw.get(key, 0)
        if type(value) is not int or value < 0 or value > _MAX_STORED:
            return None
        row[key] = value
    return row


SLEEP_REPLY = "今日对话已达上限，我要睡觉了"


def dialogue_blocked(llm):
    """今日 token 或请求次数已经到顶。测试桩和没接账本的客户端不当成到顶。"""
    usage = getattr(llm, "usage", None)
    refusal = getattr(usage, "refusal", None)
    if not callable(refusal):
        return False
    try:
        reason = refusal()
    except Exception:
        return False
    return isinstance(reason, str) and bool(reason)


class TokenUsage:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def _today(self):
        return datetime.now(BEIJING).date()

    def _read(self):
        if not self.path.is_file():
            return {"limit": 0, "call_limit": 0, "days": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("用量记录无法读取") from error
        if not isinstance(raw, dict):
            raise ValueError("用量记录无法读取")
        days = raw.get("days")
        if not isinstance(days, dict):
            days = {}
        return {"limit": _bounded(raw.get("limit", 0)), "call_limit": _bounded(raw.get("call_limit", 0)), "days": days}

    def _visible(self, data):
        today = self._today()
        found = {}
        for key, raw in data["days"].items():
            day = _parse_day(key)
            row = _row(raw)
            if day is None or row is None:
                continue
            delta = (today - day).days
            if 0 <= delta < KEEP_DAYS:
                found[day] = row
        if today not in found:
            found[today] = _blank()
        rows = [(day, found[day]) for day in sorted(found, reverse=True)]
        return today, rows

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"limit": data["limit"], "call_limit": data.get("call_limit", 0),
                   "days": {day: row for day, row in data["days"].items() if any(row.values())}}
        fd, temporary = tempfile.mkstemp(prefix=".token-usage-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
                json.dump(payload, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _prune(self, data):
        _, rows = self._visible(data)
        data["days"] = {day.isoformat(): row for day, row in rows}

    def snapshot(self):
        with self.lock:
            data = self._read()
            today, rows = self._visible(data)
            current = next(row for day, row in rows if day == today)
            limit = data["limit"]
            call_limit = data.get("call_limit", 0)
            return {"limit": limit, "call_limit": call_limit, "today": today.isoformat(),
                    "prompt": current["prompt"], "completion": current["completion"], "total": current["total"],
                    "calls": current["calls"], "unreported": current["unreported"],
                    "reached": limit > 0 and current["total"] >= limit,
                    "calls_reached": call_limit > 0 and current["calls"] >= call_limit,
                    "days": [{"day": day.isoformat(), **row} for day, row in rows]}

    def set_limits(self, limit, call_limit):
        if type(limit) is not int or not 0 <= limit <= MAX_DAILY_LIMIT:
            raise ValueError("每日 token 上限无效")
        if type(call_limit) is not int or not 0 <= call_limit <= MAX_DAILY_LIMIT:
            raise ValueError("每日请求次数上限无效")
        with self.lock:
            data = self._read()
            data["limit"] = limit
            data["call_limit"] = call_limit
            self._prune(data)
            self._write(data)

    def refusal(self):
        """超限时返回原因。记录读不出来时返回 None，避免坏文件把对话停掉。"""
        with self.lock:
            try:
                data = self._read()
            except (OSError, ValueError):
                return None
            row = _row(data["days"].get(self._today().isoformat())) or _blank()
            reasons = []
            if data["limit"] > 0 and row["total"] >= data["limit"]:
                reasons.append("今日 token 已达上限")
            if data.get("call_limit", 0) > 0 and row["calls"] >= data["call_limit"]:
                reasons.append("今日请求次数已达上限")
            return "，".join(reasons) or None

    def allow(self):
        return self.refusal() is None

    def record(self, result):
        counts = usage_from(result)
        with self.lock:
            try:
                data = self._read()
                day = self._today().isoformat()
                row = _row(data["days"].get(day)) or _blank()
                row["calls"] = min(_MAX_STORED, row["calls"] + 1)
                if counts is None:
                    row["unreported"] = min(_MAX_STORED, row["unreported"] + 1)
                else:
                    row["prompt"] = min(_MAX_STORED, row["prompt"] + counts[0])
                    row["completion"] = min(_MAX_STORED, row["completion"] + counts[1])
                    row["total"] = min(_MAX_STORED, row["total"] + counts[2])
                data["days"][day] = row
                self._prune(data)
                self._write(data)
            except (OSError, ValueError):
                logging.getLogger("LLM").warning("token 用量没有写入")
