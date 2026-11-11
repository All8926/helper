"""通过配置接入 OpenAI 兼容的大模型接口，不记录密钥或聊天正文。"""
from __future__ import annotations

import json
import logging
import math
from ipaddress import ip_address
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


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
    """这一次思考没有写出最终 JSON 时，下一回改为直接回答。"""
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
                    "api_key", "authorization", "Authorization"}
        if reserved.intersection(extra):
            raise ValueError("extra_body 不能覆盖模型、消息、输出格式、令牌上限或认证字段")
        try:
            encoded = json.dumps(extra, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode("utf-8")) > 16384:
                raise ValueError
            # 隔离调用方后续修改，扩展参数只在显式配置时发送。
            self.extra_body = json.loads(encoded)
        except (TypeError, ValueError, RecursionError):
            raise ValueError("extra_body 必须是有效 JSON，且不能超过 16 KiB") from None

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")))

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

    def complete_json(self, system, user, image=None, *, thinking=True):
        json_instruction = "\n\n请只返回一个有效的 JSON 对象，不要使用 Markdown 代码围栏或附加说明。"
        if not self.json_mode:
            system += json_instruction
        content = user
        # 只接受本机生成的 data URL，不让接口去拉取外部图片地址。
        if image is not None:
            if not isinstance(image, str) or len(image) > 6 * 1024 * 1024:
                raise LLMError("图片内容无效或过大")
            header, separator, payload_text = image.partition(";base64,")
            if separator != ";base64," or header not in (
                    "data:image/jpeg", "data:image/png", "data:image/gif", "data:image/webp") or not payload_text:
                raise LLMError("图片内容无效或过大")
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": image}}]
        payload = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                       {"role": "user", "content": content}],
                   self.token_limit_parameter: self.max_tokens, "stream": False, **self.extra_body}
        self._apply_thinking(payload, thinking)
        # 思考开着时不发 JSON 模式，避免接口把最终答案抹成空正文。提示词里仍要求 JSON。
        if self._thinking_blocks_json_mode(payload):
            payload["messages"][0]["content"] += json_instruction
        elif self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        # 看图要上传图片，短超时很容易失败；不超过配置允许的 60 秒。
        wait = min(60, max(self.timeout, 45)) if image is not None else self.timeout
        headers = {"Content-Type": "application/json"}
        if self._worldguide:
            # The gateway rejects urllib's default User-Agent.
            headers.update({"User-Agent": "helper/1.0", "Accept": "application/json"})
        else:
            headers["Authorization"] = "Bearer " + self._api_key
        try:
            for attempt in range(2):
                request = Request(self.endpoint,
                                  data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                  headers=headers, method="POST")
                if self._worldguide:
                    request.add_unredirected_header("Authorization", "Bearer " + self._api_key)
                with urlopen(request, timeout=wait) as response:
                    raw = response.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise LLMError("大模型响应过大")
                result = json.loads(raw)
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
        except HTTPError as error:
            # 不输出响应正文或请求头，避免服务器错误回显密钥。
            if self._worldguide and error.code == 429:
                raise LLMError("WorldGuide 请求受限（HTTP 429），请稍后再试", status=429) from None
            raise LLMError(f"大模型接口 HTTP {error.code}；请核对密钥、余额和模型配置", status=error.code) from None
        except (URLError, TimeoutError, OSError):
            raise LLMError("大模型连接失败或超时，请稍后再试") from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("大模型响应格式不正确，请重试") from None
