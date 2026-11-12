"""通过配置接入 OpenAI 兼容的大模型接口，不记录密钥或聊天正文。"""
from __future__ import annotations

import json
import logging
import math
import time
from ipaddress import ip_address
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MAX_TOOL_ROUNDS = 4
TOOL_RESULT_LIMIT = 3000


def is_worldguide_url(url):
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname == "api.worldguide.online"
                and parsed.port in (None, 443))
    except ValueError:
        return False


def _object_in(text):
    """从一段文字里取出唯一需要的 JSON 对象。没有就返回 None，不把原文交给调用方。"""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        opening, separator, body = text.partition("\n")
        if separator and opening.strip().lower() in ("```", "```json"):
            text = body[:-3].strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        return data
    last = None
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped, end = 0, False, False, None
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            start = text.find("{", start + 1)
            continue
        try:
            found = json.loads(text[start:end + 1])
        except ValueError:
            found = None
        if isinstance(found, dict):
            last = found
        start = text.find("{", end + 1)
    return last


def _reasoning_text(message):
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _turn_off_thinking(payload):
    """这一次思考没有写出最终答案时，下一回改为直接回答。"""
    changed = False
    thinking = payload.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        payload["thinking"] = {"type": "disabled"}
        changed = True
    if payload.get("enable_thinking") is True:
        payload["enable_thinking"] = False
        changed = True
    if payload.get("reasoning_effort") in ("high", "medium", "xhigh", "max"):
        payload["reasoning_effort"] = "low"
        changed = True
    return changed


class LLMError(RuntimeError):
    def __init__(self, message, *, status=None):
        super().__init__(message)
        self.status = status


_JSON_INSTRUCTION = "\n\n请只返回一个有效的 JSON 对象，不要使用 Markdown 代码围栏或附加说明。"


def plain(value, limit):
    """压成一行并截断，避免外部文字把控制字符带进下一次请求。"""
    text = " ".join(str(value or "").split())
    text = "".join(ch for ch in text if ord(ch) >= 32)
    return text[:limit]


def acceptable_url(url):
    if not isinstance(url, str) or any(ch.isspace() or ord(ch) < 32 for ch in url):
        return False
    try:
        parsed = urlsplit(url)
        local = parsed.hostname == "localhost"
        if not local and parsed.hostname:
            try:
                local = ip_address(parsed.hostname).is_loopback
            except ValueError:
                pass
        valid = (parsed.scheme == "https" or (parsed.scheme == "http" and local))
        return bool(valid and parsed.hostname and parsed.port != 0 and
                    not (parsed.username or parsed.password or parsed.query or parsed.fragment))
    except ValueError:
        return False


def _api_key(raw, label):
    if not isinstance(raw, str) or len(raw) > 1024 or any(ord(ch) < 32 for ch in raw):
        raise ValueError(f"{label}.api_key 格式不正确")
    return raw.strip()


def normalize_web_search(raw):
    """缺省等于关闭。HTTP 模式启用时必须有搜索地址；密钥可留空，调用时再沿用模型密钥。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("web_search 应为 JSON 对象")
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("web_search.enabled 应为布尔值")
    provider = raw.get("provider", "http")
    if provider not in ("http", "doubao"):
        raise ValueError("web_search.provider 应为 http 或 doubao")
    base_url = raw.get("base_url", "")
    if not isinstance(base_url, str):
        raise ValueError("web_search.base_url 应为地址字符串")
    base_url = base_url.strip().rstrip("/")
    if base_url and not acceptable_url(base_url):
        raise ValueError("web_search.base_url 应为不含认证信息的 HTTPS 地址；本机回环地址可使用 HTTP")
    if enabled and provider == "http" and not base_url:
        raise ValueError("启用 HTTP 联网搜索时请填写 web_search.base_url")
    api_key = _api_key(raw.get("api_key", ""), "web_search")
    timeout = raw.get("timeout_seconds", 8)
    if type(timeout) is bool or type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 20:
        raise ValueError("web_search.timeout_seconds 应为 1–20 秒")
    count = raw.get("count", 5)
    if type(count) is not int or not 1 <= count <= 10:
        raise ValueError("web_search.count 应为 1–10 的整数")
    return {"enabled": enabled, "provider": provider, "base_url": base_url,
            "api_key": api_key, "timeout_seconds": timeout, "count": count}


def public_web_search(raw):
    """给网页的搜索配置。不带回密钥原文。"""
    try:
        value = normalize_web_search(raw)
    except ValueError:
        value = normalize_web_search(None)
        if isinstance(raw, dict):
            if raw.get("provider") in ("http", "doubao"):
                value["provider"] = raw["provider"]
            if isinstance(raw.get("base_url"), str):
                value["base_url"] = raw["base_url"].strip().rstrip("/")
            if raw.get("enabled") is True:
                value["enabled"] = True
            if type(raw.get("count")) is int and 1 <= raw["count"] <= 10:
                value["count"] = raw["count"]
            if type(raw.get("timeout_seconds")) in (int, float) and not isinstance(raw.get("timeout_seconds"), bool):
                value["timeout_seconds"] = raw["timeout_seconds"]
            value["api_key"] = raw.get("api_key") if isinstance(raw.get("api_key"), str) else ""
    shown = {key: value[key] for key in ("enabled", "provider", "base_url", "timeout_seconds", "count")}
    shown["api_key_configured"] = bool(value.get("api_key"))
    return shown


def normalize_vision(raw):
    """看图模型。model 留空时直接用对话模型看图；地址和密钥留空时沿用对话模型的。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("vision 应为 JSON 对象")
    base_url = raw.get("base_url", "")
    if not isinstance(base_url, str):
        raise ValueError("vision.base_url 应为地址字符串")
    base_url = base_url.strip().rstrip("/")
    if base_url and not acceptable_url(base_url):
        raise ValueError("vision.base_url 应为不含认证信息的 HTTPS 地址；本机回环地址可使用 HTTP")
    model = raw.get("model", "")
    if not isinstance(model, str) or len(model) > 200:
        raise ValueError("vision.model 应为模型名")
    return {"base_url": base_url, "api_key": _api_key(raw.get("api_key", ""), "vision"), "model": model.strip()}


def public_vision(raw):
    try:
        value = normalize_vision(raw)
    except ValueError:
        value = normalize_vision(None)
    return {"base_url": value["base_url"], "model": value["model"], "api_key_configured": bool(value["api_key"])}


def post_json(url, payload, timeout, api_key, *, label="大模型"):
    """发送一次 JSON 并解析响应。不记录密钥和正文，错误信息里也不带响应原文。"""
    worldguide = is_worldguide_url(url)
    # 部分网关会拦截 urllib 默认的 User-Agent。
    headers = {"Content-Type": "application/json", "User-Agent": "helper/1.0"}
    if worldguide:
        headers["Accept"] = "application/json"
    else:
        headers["Authorization"] = "Bearer " + api_key
    try:
        request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers=headers, method="POST")
        if worldguide:
            request.add_unredirected_header("Authorization", "Bearer " + api_key)
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise LLMError(f"{label}响应过大")
        return json.loads(raw)
    except HTTPError as error:
        if worldguide and error.code == 429:
            noun = "WorldGuide" if label == "大模型" else label
            raise LLMError(f"{noun} 请求受限（HTTP 429），请稍后再试", status=429) from None
        if label == "大模型":
            raise LLMError(f"大模型接口 HTTP {error.code}；请核对密钥、余额和模型配置", status=error.code) from None
        raise LLMError(f"{label} HTTP {error.code}", status=error.code) from None
    except LLMError:
        raise
    except (URLError, TimeoutError, OSError):
        raise LLMError(f"{label}连接失败或超时，请稍后再试") from None
    except (ValueError, TypeError):
        raise LLMError(f"{label}响应格式不正确，请重试") from None


def _choice_message(result):
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except (TypeError, KeyError, IndexError):
        raise LLMError("大模型响应格式不正确，请重试") from None
    if not isinstance(message, dict):
        raise LLMError("大模型响应格式不正确，请重试")
    finish = choice.get("finish_reason")
    if finish == "length":
        raise LLMError("大模型响应被截断，请增加 max_tokens")
    if finish == "content_filter" or message.get("refusal"):
        raise LLMError("大模型接口未提供可用正文，请检查服务商的内容限制")
    return message, finish


def _tool_calls(message):
    raw = message.get("tool_calls") or message.get("function_call")
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or len(raw) > 8:
        raise LLMError("大模型返回的工具调用无效")
    calls = []
    for tool in raw:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            raise LLMError("大模型返回的工具调用无效")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name:
            raise LLMError("大模型返回的工具调用无效")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        if not isinstance(arguments, str):
            arguments = ""
        call_id = tool.get("id") if isinstance(tool.get("id"), str) else ""
        if not call_id:
            raise LLMError("大模型返回的工具调用无效")
        calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
    return calls


def _assistant_tool_message(message, payload, calls):
    """带工具的下一轮必须带回思考内容，否则官方 DeepSeek 会返回 400。"""
    assistant = {"role": "assistant", "content": message.get("content") or ""}
    reasoning = message.get("reasoning_content")
    thinking = payload.get("thinking")
    if isinstance(reasoning, str) or (isinstance(thinking, dict) and thinking.get("type") == "enabled"):
        assistant["reasoning_content"] = reasoning if isinstance(reasoning, str) else ""
    assistant["tool_calls"] = calls
    return assistant


def _run_tool(tools, call):
    """工具出错时把错误交给模型，不中断这次回复。"""
    log = logging.getLogger("LLM")
    name = call["function"]["name"]
    tool = tools.get(name)
    if tool is None:
        return "没有这个工具"
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except ValueError:
        args = None
    if not isinstance(args, dict):
        return "工具参数应为 JSON 对象"
    started = time.perf_counter()
    try:
        text = str(tool.run(args))
    except (LLMError, ValueError) as error:
        text = f"工具出错：{error}"
    except Exception:
        log.exception("工具 | %s | 执行出错", name)
        text = "工具出错"
    log.info("工具 | %s | %.2f秒 | 结果字数=%d", name, time.perf_counter() - started, len(text))
    return text[:TOOL_RESULT_LIMIT]


class LLMClient:
    _DEEPSEEK_THINKING_MODELS = frozenset((
        "deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat"))

    def __init__(self, options):
        if not isinstance(options, dict):
            raise ValueError("大模型配置必须是 JSON 对象")
        base_url = options.get("base_url", "https://api.deepseek.com")
        if not isinstance(base_url, str):
            raise ValueError("大模型 base_url 必须是接口地址字符串")
        self.base_url = base_url.rstrip("/")
        try:
            parsed = urlsplit(self.base_url)
            local = parsed.hostname == "localhost"
            if not local and parsed.hostname:
                try:
                    local = ip_address(parsed.hostname).is_loopback
                except ValueError:
                    pass
            valid = (parsed.scheme == "https" or (parsed.scheme == "http" and local))
            valid = valid and parsed.hostname and parsed.port != 0
            valid = valid and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        except ValueError:
            valid = False
        if not valid or any(ch.isspace() or ord(ch) < 32 for ch in base_url):
            raise ValueError("大模型 base_url 应为不含认证信息的 HTTPS 地址；本机回环地址可使用 HTTP")
        self.endpoint = (self.base_url if self.base_url.endswith("/chat/completions")
                         else self.base_url + "/chat/completions")
        self._deepseek_official = (parsed.scheme == "https" and parsed.hostname == "api.deepseek.com"
                                  and parsed.port in (None, 443))
        self._worldguide = is_worldguide_url(self.base_url)
        self.model = options.get("model", "deepseek-flash")
        self._api_key = options.get("api_key", "")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("大模型 model 不能为空")
        if not isinstance(self._api_key, str) or not self._api_key.strip():
            raise ValueError("请在大模型配置文件中填写 api_key")
        self.timeout = options.get("timeout_seconds", 20)
        self.max_tokens = options.get("max_tokens", 512)
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or not 1 <= self.timeout <= 60:
            raise ValueError("timeout_seconds 应为 1–60 秒")
        if type(self.max_tokens) is not int or not 128 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens 应为 128–4096 的整数")
        self.json_mode = options.get("json_mode", True)
        if type(self.json_mode) is not bool:
            raise ValueError("json_mode 应为布尔值")
        self.token_limit_parameter = options.get("token_limit_parameter", "max_tokens")
        if self.token_limit_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("token_limit_parameter 应为 max_tokens 或 max_completion_tokens")
        extra = options.get("extra_body", {})
        if not isinstance(extra, dict) or not all(isinstance(key, str) for key in extra):
            raise ValueError("extra_body 应为 JSON 对象")
        reserved = {"model", "messages", "stream", "response_format", "max_tokens", "max_completion_tokens",
                    "tools", "tool_choice", "web_search_options", "api_key", "authorization", "Authorization"}
        if reserved.intersection(extra):
            raise ValueError("extra_body 不能覆盖模型、消息、输出格式、令牌上限、工具或认证字段")
        try:
            encoded = json.dumps(extra, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode("utf-8")) > 16384:
                raise ValueError
            # 隔离调用方后续修改，扩展参数只在显式配置时发送。
            self.extra_body = json.loads(encoded)
        except (TypeError, ValueError, RecursionError):
            raise ValueError("extra_body 必须是有效 JSON，且不能超过 16 KiB") from None
        from .decision import normalize_decision
        self.web_search = normalize_web_search(options.get("web_search"))
        self.vision = normalize_vision(options.get("vision"))
        self.decision = normalize_decision(options.get("decision"))
        self._options = options

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    @property
    def api_key(self):
        return self._api_key

    def vision_client(self):
        """看图用的客户端。没单独配置看图模型时就是自己。"""
        if not self.vision["model"]:
            return self
        return LLMClient({**self._options, "model": self.vision["model"],
                          "base_url": self.vision["base_url"] or self.base_url,
                          "api_key": self.vision["api_key"] or self._api_key,
                          "extra_body": {}, "web_search": None, "vision": None, "decision": None})

    def _apply_thinking(self, payload, enabled):
        """按群开关写各家的思考字段。扩展参数里已经写过的项不覆盖。"""
        name = self.model.lower()
        extra = self.extra_body
        if "qwen" in name or name.startswith("qwq"):
            if "enable_thinking" not in extra:
                payload["enable_thinking"] = bool(enabled)
            return
        if "grok" in name:
            # Grok 4.5 及以后不能完全关闭思考，关掉时只用最低强度。
            if not enabled:
                payload["reasoning_effort"] = "low"
            elif "reasoning_effort" not in extra:
                payload["reasoning_effort"] = "high"
            return
        if "doubao" in name or name.startswith("seed"):
            if "thinking" not in extra:
                payload["thinking"] = {"type": "enabled" if enabled else "disabled"}
            return
        if (self._deepseek_official and name in self._DEEPSEEK_THINKING_MODELS
                and not {"thinking", "reasoning_effort"}.intersection(extra)):
            payload["thinking"] = {"type": "enabled" if enabled else "disabled"}

    def _thinking_blocks_json_mode(self, payload):
        """官方 DeepSeek 思考开着时不发 JSON 模式。该模式会把认不出的正文换成空格。"""
        if not self.json_mode or not self._deepseek_official:
            return False
        if self.model.lower() not in self._DEEPSEEK_THINKING_MODELS:
            return False
        thinking = payload.get("thinking")
        return not (isinstance(thinking, dict) and thinking.get("type") == "disabled")

    def _direct_answer(self, payload):
        """思考没有交出 JSON 时，关掉思考并恢复 JSON 模式。"""
        changed = _turn_off_thinking(payload)
        if (self._deepseek_official and self.model.lower() in self._DEEPSEEK_THINKING_MODELS
                and "thinking" not in self.extra_body):
            thinking = payload.get("thinking")
            if not (isinstance(thinking, dict) and thinking.get("type") == "disabled"):
                payload["thinking"] = {"type": "disabled"}
                changed = True
        if changed and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return changed

    @staticmethod
    def _user_content(user, image):
        # 只接受本机生成的 data URL，不让接口去拉取外部图片地址。
        if image is None:
            return user
        if not isinstance(image, str) or len(image) > 6 * 1024 * 1024:
            raise LLMError("图片内容无效或过大")
        header, separator, payload_text = image.partition(";base64,")
        if separator != ";base64," or header not in (
                "data:image/jpeg", "data:image/png", "data:image/gif", "data:image/webp") or not payload_text:
            raise LLMError("图片内容无效或过大")
        return [{"type": "text", "text": user}, {"type": "image_url", "image_url": {"url": image}}]

    def _base_payload(self, system, user, image, thinking):
        payload = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                       {"role": "user", "content": self._user_content(user, image)}],
                   self.token_limit_parameter: self.max_tokens, "stream": False, **self.extra_body}
        self._apply_thinking(payload, thinking)
        # 看图要上传图片，短超时很容易失败；不超过配置允许的 60 秒。
        wait = min(60, max(self.timeout, 45)) if image is not None else self.timeout
        return payload, wait

    def _post_json(self, url, payload, timeout, api_key):
        return post_json(url, payload, timeout, api_key)

    def chat(self, system, user, tools=(), *, image=None, thinking=True):
        """纯文本回复。模型自己决定调不调工具，最多 MAX_TOOL_ROUNDS 轮，之后只要正文。"""
        log = logging.getLogger("LLM")
        tools = {tool.name: tool for tool in tools}
        payload, wait = self._base_payload(system, user, image, thinking)
        if tools:
            payload["tools"] = [tool.spec() for tool in tools.values()]
        rounds, retried = 0, False
        while True:
            try:
                result = self._post_json(self.endpoint, payload, wait, self._api_key)
            except LLMError as error:
                # 不支持工具的模型会直接报 400，去掉工具再问一次。
                if error.status == 400 and "tools" in payload and rounds == 0:
                    log.warning("模型接口不接受工具，改为不带工具再请求一次")
                    payload.pop("tools")
                    tools = {}
                    continue
                raise
            message, finish = _choice_message(result)
            calls = _tool_calls(message) if tools else []
            if calls:
                rounds += 1
                payload["messages"].append(_assistant_tool_message(message, payload, calls))
                for call in calls:
                    payload["messages"].append({"role": "tool", "tool_call_id": call["id"],
                                                "content": _run_tool(tools, call)})
                if rounds >= MAX_TOOL_ROUNDS:
                    payload["tool_choice"] = "none"
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if not retried and finish in (None, "stop"):
                retried = True
                if _reasoning_text(message) and _turn_off_thinking(payload):
                    log.warning("深度思考没有给出最终答案，改为直接回答并重试一次")
                else:
                    log.warning("大模型返回空正文，按原请求重试一次")
                continue
            suffix = "（只收到思考内容，未取得最终答案）" if _reasoning_text(message) else ""
            raise LLMError("大模型返回了空结果" + suffix)

    def complete_json(self, system, user, image=None, *, thinking=True):
        payload, wait = self._base_payload(system if self.json_mode else system + _JSON_INSTRUCTION,
                                           user, image, thinking)
        # 思考开着时不发 JSON 模式，避免接口把最终答案抹成空正文。提示词里仍要求 JSON。
        if self._thinking_blocks_json_mode(payload):
            payload["messages"][0]["content"] += _JSON_INSTRUCTION
        elif self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            for attempt in range(2):
                result = self._post_json(self.endpoint, payload, wait, self._api_key)
                choice = result["choices"][0]
                finish = choice.get("finish_reason")
                message = choice["message"]
                if finish == "length":
                    raise LLMError("大模型响应被截断，请增加 max_tokens")
                if finish == "content_filter" or message.get("refusal"):
                    raise LLMError("大模型接口未提供可用正文，请检查服务商的内容限制")
                if finish in ("tool_calls", "function_call") or message.get("tool_calls") or message.get("function_call"):
                    raise LLMError("大模型返回了工具调用，当前功能需要直接返回 JSON 正文")
                content = message.get("content")
                blank = content is None or (isinstance(content, str) and not content.strip())
                if blank:
                    reasoning = _reasoning_text(message)
                    found = _object_in(reasoning)
                    if found is not None:
                        return found
                    # 只有思考过程、没有最终 JSON 时，关掉深度思考再要一次正文。
                    # 没有思考内容时仍按同一次请求重试，避免把偶发空正文误判成思考问题。
                    if attempt == 0 and finish in (None, "stop"):
                        if reasoning and self._direct_answer(payload):
                            logging.getLogger("LLM").warning("深度思考没有给出最终答案，改为直接回答并重试一次")
                        else:
                            logging.getLogger("LLM").warning("大模型返回空正文，按原请求重试一次")
                        continue
                    suffix = "（只收到思考内容，未取得最终答案）" if reasoning else ""
                    raise LLMError("大模型返回了空结果" + suffix)
                if not isinstance(content, str):
                    raise LLMError("大模型正文格式不正确")
                content = content.strip()
                relaxed = self._thinking_blocks_json_mode(payload)
                # JSON 模式关闭，或官方思考请求未发 JSON 模式时，接受围栏里的 JSON。
                if "response_format" not in payload and content.startswith("```") and content.endswith("```"):
                    opening, separator, body = content.partition("\n")
                    if separator and opening.strip().lower() in ("```", "```json"):
                        content = body[:-3].strip()
                invalid = False
                try:
                    data = json.loads(content)
                except ValueError:
                    invalid = True
                    data = _object_in(content) if relaxed else None
                if isinstance(data, dict):
                    return data
                if relaxed and attempt == 0 and finish in (None, "stop") and self._direct_answer(payload):
                    logging.getLogger("LLM").warning("深度思考没有返回 JSON，改为直接回答并重试一次")
                    continue
                if invalid:
                    raise LLMError("大模型没有返回 JSON 对象，请重试")
                raise LLMError("大模型结果应为 JSON 对象")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("大模型响应格式不正确，请重试") from None
