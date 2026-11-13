"""文本发送前的可配置随机等待；仅控制节奏，不保证账号风控结果。"""
from __future__ import annotations

import contextvars
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSendDelay:
    enabled: bool = True
    characters_per_group: int = 10
    min_seconds: float = 0.1
    max_seconds: float = 0.3

    @classmethod
    def from_config(cls, options=None):
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise ValueError("text_send_delay 必须是 JSON 对象")
        result = cls(enabled=options.get("enabled", True),
                     characters_per_group=options.get("characters_per_group", 10),
                     min_seconds=options.get("min_seconds", 0.1),
                     max_seconds=options.get("max_seconds", 0.3))
        if type(result.enabled) is not bool:
            raise ValueError("text_send_delay.enabled 必须是布尔值")
        if type(result.characters_per_group) is not int or not 1 <= result.characters_per_group <= 10000:
            raise ValueError("text_send_delay.characters_per_group 应为 1–10000 的整数")
        for value in (result.min_seconds, result.max_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 60:
                raise ValueError("text_send_delay 的等待秒数应为 0–60 的有限数值")
        if result.min_seconds > result.max_seconds:
            raise ValueError("text_send_delay.min_seconds 不能大于 max_seconds")
        return result

    def seconds_for(self, text):
        if not self.enabled:
            return 0.0
        # 按 Unicode 字符计数；不足一组向上取整，每组独立抽取等待时间。
        groups = (len(text) + self.characters_per_group - 1) // self.characters_per_group
        return sum(random.uniform(self.min_seconds, self.max_seconds) for _ in range(groups))


class SendClock:
    """一次回复里，文字等待、语音合成和微信发送各自用了多久。"""

    def __init__(self):
        self.wait = 0.0
        self.synth = 0.0
        self.native = 0.0


_clock = contextvars.ContextVar("helper_send_clock", default=None)


@contextmanager
def counting_send():
    clock = SendClock()
    token = _clock.set(clock)
    try:
        yield clock
    finally:
        _clock.reset(token)


def note_wait(seconds):
    clock = _clock.get()
    if clock is not None:
        clock.wait += seconds


def note_synth(seconds):
    clock = _clock.get()
    if clock is not None:
        clock.synth += seconds


def note_native(seconds):
    clock = _clock.get()
    if clock is not None:
        clock.native += seconds


def describe_send(clock, total):
    """把发送总时长拆成能看懂的几段。不足 0.01 秒的段不写。"""
    parts = []
    if clock.wait >= 0.01:
        parts.append(f"文字等待 {clock.wait:.2f}秒")
    if clock.synth >= 0.01:
        parts.append(f"语音合成 {clock.synth:.2f}秒")
    if clock.native >= 0.01:
        parts.append(f"微信发送 {clock.native:.2f}秒")
    rest = total - clock.wait - clock.synth - clock.native
    if rest >= 0.05 or not parts:
        parts.append(f"其余 {max(rest, 0):.2f}秒")
    return " | ".join(parts)
