"""决策模型：只回答程序里的是非题，例如现在该不该说话、这句要不要念出来。

没配置或调用失败时，按页面上为每种决策配置的概率随机给出是或否。"""
from __future__ import annotations

import json
import logging
import math
import random

from .llm import LLMError, acceptable_url, post_json

DEFAULT_BASE_URL = "https://emrank.tech/v1/decisions"
# 早期默认只写到 /v1，路径由代码补上。现在地址原样使用，读到这一条旧值时改成完整地址。
_LEGACY_BASE_URL = "https://emrank.tech/v1"
DEFAULT_MODEL = "typesafe/jev[slow]"
DECISIONS = ("speak", "voice", "sing")
DEFAULT_FALLBACK = {"speak": 0.3, "voice": 0.3, "sing": 0.0}
STATE_LIMIT = 12000


def _probability(value, label):
    if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{label} 应为 0–1 之间的数")
    return float(value)


def normalize_decision(raw):
    """缺省等于关闭。启用时必须有地址、密钥和模型名。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("decision 应为 JSON 对象")
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("decision.enabled 应为布尔值")
    base_url = raw.get("base_url", DEFAULT_BASE_URL)
    if not isinstance(base_url, str):
        raise ValueError("decision.base_url 应为地址字符串")
    base_url = base_url.strip().rstrip("/") or DEFAULT_BASE_URL
    if base_url == _LEGACY_BASE_URL:
        base_url = DEFAULT_BASE_URL
    if not acceptable_url(base_url):
        raise ValueError("decision.base_url 应为不含认证信息的 HTTPS 地址；本机回环地址可使用 HTTP")
    api_key = raw.get("api_key", "")
    if not isinstance(api_key, str) or len(api_key) > 1024 or any(ord(ch) < 32 for ch in api_key):
        raise ValueError("decision.api_key 格式不正确")
    model = raw.get("model", DEFAULT_MODEL)
    if not isinstance(model, str) or len(model) > 200:
        raise ValueError("decision.model 应为模型名")
    model = model.strip() or DEFAULT_MODEL
    if enabled and not api_key.strip():
        raise ValueError("启用决策模型时请填写 decision.api_key")
    timeout = raw.get("timeout_seconds", 20)
    if type(timeout) is bool or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 60:
        raise ValueError("decision.timeout_seconds 应为 1–60 秒")
    fallback = raw.get("fallback", {})
    if not isinstance(fallback, dict):
        raise ValueError("decision.fallback 应为 JSON 对象")
    chances = {name: _probability(fallback.get(name, DEFAULT_FALLBACK[name]), f"decision.fallback.{name}")
               for name in DECISIONS}
    return {"enabled": enabled, "base_url": base_url, "api_key": api_key.strip(), "model": model,
            "timeout_seconds": timeout, "fallback": chances}


def public_decision(raw):
    """给网页的决策模型配置。不带回密钥原文。"""
    try:
        value = normalize_decision(raw)
    except ValueError:
        value = normalize_decision(None)
        value["api_key"] = raw.get("api_key") if isinstance(raw, dict) and isinstance(raw.get("api_key"), str) else ""
    shown = {key: value[key] for key in ("enabled", "base_url", "model", "timeout_seconds", "fallback")}
    shown["api_key_configured"] = bool(value["api_key"])
    return shown


class Decider:
    def __init__(self, options, *, logger=None):
        self.options = normalize_decision(options)
        self.logger = logger or logging.getLogger("Decision")

    def yes(self, name, instructions, criteria, state, *, threshold=0.5):
        """问一道是非题。返回 True 表示“是”。"""
        if name not in DECISIONS:
            raise ValueError("未知的决策名称")
        probability = self._ask(name, instructions, criteria, state)
        if probability is None:
            chance = self.options["fallback"][name]
            answer = random.random() < chance
            self.logger.info("决策 | %s | 决策模型不可用，按 %.0f%% 概率随机：%s",
                             name, chance * 100, "是" if answer else "否")
            return answer
        answer = probability >= threshold
        self.logger.info("决策 | %s | 概率 %.2f | 阈值 %.2f | %s", name, probability, threshold,
                         "是" if answer else "否")
        return answer

    def _ask(self, name, instructions, criteria, state):
        """返回 0–1 的概率。未启用或失败时返回 None。"""
        if not self.options["enabled"]:
            return None
        encoded = json.dumps(state, ensure_ascii=False)
        # 带聊天记录的决策保持每一条原文。没有记录、又长过上限时，只留末尾。
        if len(encoded) > STATE_LIMIT and not (isinstance(state, dict) and isinstance(state.get("transcript"), list)):
            state = {"context": encoded[-STATE_LIMIT:]}
        body = {"model": self.options["model"], "state": state,
                "questions": {name: {"type": "noul", "instructions": instructions,
                                     "criteria": {"true": criteria[0], "false": criteria[1]}}}}
        try:
            result = post_json(self.options["base_url"], body, self.options["timeout_seconds"],
                               self.options["api_key"], label="决策模型")
            value = result["answers"][name]["noul"]
        except LLMError as error:
            self.logger.warning("决策 | %s | %s", name, error)
            return None
        except (TypeError, KeyError):
            self.logger.warning("决策 | %s | 决策模型响应格式不正确", name)
            return None
        if type(value) is bool or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            self.logger.warning("决策 | %s | 决策模型返回的概率无效", name)
            return None
        return float(value)
