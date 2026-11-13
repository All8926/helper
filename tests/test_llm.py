"""大模型协议和错误处理验证；使用假响应，不消耗实际 API。"""
import base64
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from PIL import Image

from wxbot_framework.chat_tools import Tool, chat_tools, web_search_tool
from wxbot_framework.decision import Decider, normalize_decision, public_decision
from wxbot_framework.llm import MAX_TOOL_ROUNDS, LLMClient, LLMError
from wxbot_framework.vision import SeenImages, vision_data_url
from plugins.idiom_chain.judge import IdiomJudge


class JSONClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def complete_json(self, system, user):
        self.calls.append((system, user))
        return self.result


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.options = {"api_key": "dummy-test-key", "model": "deepseek-flash"}
        self.client = LLMClient(self.options)

    def response(self, content, **extra):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, **extra}]}).encode())

    def test_official_json_request_and_utf8(self):
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"idiom":"一心一意"}')) as call:
            self.assertEqual(self.client.complete_json("只返回 JSON", "中文"), {"idiom": "一心一意"})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["messages"][1]["content"], "中文")
        self.assertNotIn("response_format", payload)
        self.assertIn("JSON 对象", payload["messages"][0]["content"])
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(call.call_args.kwargs["timeout"], 20)

    def test_v4_flash_defaults_to_deep_thinking(self):
        client = LLMClient({**self.options, "model": "deepseek-v4-flash"})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response("{}")) as call:
            client.complete_json("JSON", "测试")
        self.assertEqual(json.loads(call.call_args.args[0].data)["thinking"], {"type": "enabled"})
        self.assertNotIn("response_format", json.loads(call.call_args.args[0].data))

    def test_group_can_turn_deep_thinking_off(self):
        with patch("wxbot_framework.llm.urlopen", return_value=self.response("{}")) as call:
            self.client.complete_json("JSON", "测试", thinking=False)
        self.assertEqual(json.loads(call.call_args.args[0].data)["thinking"], {"type": "disabled"})
        self.assertEqual(json.loads(call.call_args.args[0].data)["response_format"], {"type": "json_object"})

    def test_doubao_qwen_and_grok_follow_the_same_switch(self):
        cases = (
            ("doubao-seed-2-0-lite", True, "thinking", {"type": "enabled"}),
            ("doubao-seed-2-0-lite", False, "thinking", {"type": "disabled"}),
            ("qwen3.8-max", True, "enable_thinking", True),
            ("qwen3.8-max", False, "enable_thinking", False),
            ("grok-4.7", True, "reasoning_effort", "high"),
            ("grok-4.7", False, "reasoning_effort", "low"),
        )
        for model, enabled, field, value in cases:
            with self.subTest(model=model, enabled=enabled):
                client = LLMClient({**self.options, "base_url": "https://gateway.example.test/v1", "model": model})
                with patch("wxbot_framework.llm.urlopen", return_value=self.response("{}")) as call:
                    client.complete_json("JSON", "测试", thinking=enabled)
                self.assertEqual(json.loads(call.call_args.args[0].data)[field], value)

    def test_compatible_gateway_preserves_version_path_without_vendor_parameters(self):
        self.client = LLMClient({**self.options, "base_url": "https://gateway.example.test/v1/", "model": "custom-model"})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"reply":"你好"}')) as call:
            self.assertEqual(self.client.complete_json("返回 JSON", "你好"), {"reply": "你好"})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://gateway.example.test/v1/chat/completions")
        self.assertEqual(set(json.loads(request.data)), {"model", "messages", "response_format", "max_tokens", "stream"})

    def test_full_endpoint_is_not_appended_twice(self):
        client = LLMClient({**self.options, "base_url": "https://gateway.example.test/v1/chat/completions/"})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{}')) as call:
            client.complete_json("JSON", "测试")
        self.assertEqual(call.call_args.args[0].full_url, "https://gateway.example.test/v1/chat/completions")

    def test_worldguide_grok_uses_gateway_headers_and_scopes_credentials(self):
        client = LLMClient({**self.options, "base_url": "https://api.worldguide.online/v1",
                            "model": "grok-4.7", "extra_body": {"reasoning_effort": "low"}})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"reply":"你好"}')) as call:
            self.assertEqual(client.complete_json("返回 JSON", "你好"), {"reply": "你好"})
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.worldguide.online/v1/chat/completions")
        self.assertEqual(request.get_header("User-agent"), "helper/1.0")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(request.get_header("Authorization"), "Bearer dummy-test-key")
        self.assertNotIn("Authorization", request.headers)
        payload = json.loads(request.data)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("thinking", payload)

    def test_completion_token_parameter_sends_only_selected_limit(self):
        client = LLMClient({**self.options, "token_limit_parameter": "max_completion_tokens", "max_tokens": 2048})
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{}')) as call:
            client.complete_json("JSON", "测试")
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", body)

    def test_prompt_json_mode_omits_response_format_and_accepts_fenced_json(self):
        client = LLMClient({**self.options, "json_mode": False})
        for content in ('{"reply":"你好"}', '```json\n{"reply":"你好"}\n```', '```\n{"reply":"你好"}\n```'):
            with self.subTest(content=content), patch("wxbot_framework.llm.urlopen", return_value=self.response(content)) as call:
                self.assertEqual(client.complete_json("原功能规则", "测试"), {"reply": "你好"})
                body = json.loads(call.call_args.args[0].data)
                self.assertNotIn("response_format", body)
                self.assertIn("原功能规则", body["messages"][0]["content"])
                self.assertIn("JSON 对象", body["messages"][0]["content"])

    def test_prompt_mode_still_requires_a_single_json_object(self):
        client = LLMClient({**self.options, "json_mode": False})
        for content in ('这是普通回答', '[]', '前言\n```json\n{}\n```', '```json\n{}\n```\n后记', '{} {}'):
            with self.subTest(content=content), patch("wxbot_framework.llm.urlopen", return_value=self.response(content)):
                with self.assertRaises(LLMError):
                    client.complete_json("JSON", "测试")

    def test_vendor_parameters_are_opt_in_and_copied(self):
        extra = {"thinking": {"type": "disabled"}, "temperature": 0.5}
        client = LLMClient({**self.options, "extra_body": extra})
        extra["thinking"]["type"] = "enabled"
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{}')) as call:
            client.complete_json("JSON", "测试")
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["temperature"], 0.5)

    def test_explicit_reasoning_options_override_official_defaults(self):
        for extra in ({"thinking":{"type":"enabled"}}, {"reasoning_effort":"low"}):
            with self.subTest(extra=extra):
                client = LLMClient({**self.options, "extra_body":extra})
                with patch("wxbot_framework.llm.urlopen", return_value=self.response('{}')) as call:
                    client.complete_json("JSON", "测试")
                body = json.loads(call.call_args.args[0].data)
                for key, value in extra.items():
                    self.assertEqual(body[key], value)
                if extra.get("thinking", {}).get("type") == "disabled":
                    self.assertEqual(body["response_format"], {"type": "json_object"})
                else:
                    self.assertNotIn("response_format", body)
                if "thinking" not in extra:
                    self.assertNotIn("thinking", body)

    def test_legacy_reasoner_and_similar_host_do_not_receive_non_thinking_default(self):
        for change in ({"model":"deepseek-reasoner"}, {"base_url":"https://api.deepseek.com.example.test/v1"}):
            with self.subTest(change=change):
                client = LLMClient({**self.options, **change})
                with patch("wxbot_framework.llm.urlopen", return_value=self.response('{}')) as call:
                    client.complete_json("JSON", "测试")
                self.assertNotIn("thinking", json.loads(call.call_args.args[0].data))

    def test_official_empty_json_response_retries_the_same_json_request(self):
        with patch("wxbot_framework.llm.urlopen", side_effect=[self.response(''), self.response('{"reply":"成功"}')]) as call:
            self.assertEqual(self.client.complete_json("原功能规则 JSON", "原输入"), {"reply": "成功"})
        self.assertEqual(call.call_count, 2)
        first, second = [json.loads(args.args[0].data) for args in call.call_args_list]
        self.assertEqual(first, second)
        self.assertNotIn("response_format", first)
        self.assertEqual(first["thinking"], {"type": "enabled"})
        self.assertIn("JSON 对象", first["messages"][0]["content"])
        self.assertTrue(self.client.json_mode)

    def test_gateway_empty_response_retry_keeps_configured_protocol(self):
        client = LLMClient({**self.options,"base_url":"https://gateway.example.test/v1"})
        with patch("wxbot_framework.llm.urlopen", side_effect=[self.response(None),self.response('{"reply":"成功"}')]) as call:
            self.assertEqual(client.complete_json("JSON", "测试"), {"reply":"成功"})
        requests = [json.loads(args.args[0].data) for args in call.call_args_list]
        self.assertEqual(requests[0], requests[1])
        self.assertNotIn("thinking", requests[0])

    def test_reasoning_only_responses_never_become_the_reply_and_retry_is_bounded(self):
        raw = {"choices":[{"finish_reason":"stop","message":{"content":"", "reasoning_content":"private-reasoning-text"}}]}
        with patch("wxbot_framework.llm.urlopen", side_effect=[io.BytesIO(json.dumps(raw).encode()),io.BytesIO(json.dumps(raw).encode())]) as call:
            with self.assertRaisesRegex(LLMError,"未取得最终答案") as caught:
                self.client.complete_json("JSON", "测试")
        self.assertEqual(call.call_count, 2)
        first, second = [json.loads(args.args[0].data) for args in call.call_args_list]
        self.assertEqual(first["thinking"], {"type": "enabled"})
        self.assertNotIn("response_format", first)
        self.assertEqual(second["thinking"], {"type": "disabled"})
        self.assertEqual(second["response_format"], {"type": "json_object"})
        self.assertNotIn("private-reasoning-text", str(caught.exception))

    def test_json_left_in_reasoning_is_used_without_another_request(self):
        client = LLMClient({**self.options, "model": "deepseek-v4-flash"})
        raw = {"choices":[{"finish_reason":"stop","message":{
            "content": "",
            "reasoning_content": "先组织答案。\n{\"reply\":\"可以\"}"}}]}
        with patch("wxbot_framework.llm.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as call:
            self.assertEqual(client.complete_json("JSON", "测试"), {"reply": "可以"})
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("response_format", json.loads(call.call_args.args[0].data))

    def test_thinking_request_reads_json_wrapped_in_text(self):
        client = LLMClient({**self.options, "model": "deepseek-v4-flash"})
        raw = {"choices":[{"finish_reason":"stop","message":{"content":"结果如下\n{\"reply\":\"可以\"}"}}]}
        with patch("wxbot_framework.llm.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as call:
            self.assertEqual(client.complete_json("只返回 JSON", "测试"), {"reply": "可以"})
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("response_format", body)
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertIn("JSON 对象", body["messages"][0]["content"])

    def test_non_json_text_retries_with_thinking_off_and_json_mode(self):
        client = LLMClient({**self.options, "model": "deepseek-v4-flash"})
        first_raw = {"choices":[{"finish_reason":"stop","message":{"content":"先不给结构"}}]}
        second_raw = {"choices":[{"finish_reason":"stop","message":{"content":"{\"reply\":\"成功\"}"}}]}
        bodies = [io.BytesIO(json.dumps(first_raw).encode()), io.BytesIO(json.dumps(second_raw).encode())]
        with patch("wxbot_framework.llm.urlopen", side_effect=bodies) as call:
            self.assertEqual(client.complete_json("JSON", "测试"), {"reply": "成功"})
        first, second = [json.loads(args.args[0].data) for args in call.call_args_list]
        self.assertNotIn("response_format", first)
        self.assertEqual(first["thinking"], {"type": "enabled"})
        self.assertEqual(second["thinking"], {"type": "disabled"})
        self.assertEqual(second["response_format"], {"type": "json_object"})

    def test_reasoning_without_json_retries_once_with_thinking_disabled(self):
        client = LLMClient({**self.options, "model": "deepseek-v4-flash"})
        first_raw = {"choices":[{"finish_reason":"stop","message":{"content":None,"reasoning_content":"private-chain"}}]}
        second_raw = {"choices":[{"finish_reason":"stop","message":{"content":"{\"reply\":\"成功\"}"}}]}
        bodies = [io.BytesIO(json.dumps(first_raw).encode()), io.BytesIO(json.dumps(second_raw).encode())]
        with patch("wxbot_framework.llm.urlopen", side_effect=bodies) as call:
            self.assertEqual(client.complete_json("JSON", "测试"), {"reply": "成功"})
        first, second = [json.loads(args.args[0].data) for args in call.call_args_list]
        self.assertEqual(first["thinking"], {"type": "enabled"})
        self.assertNotIn("response_format", first)
        self.assertEqual(second["thinking"], {"type": "disabled"})
        self.assertEqual(second["response_format"], {"type": "json_object"})

    def test_doubao_qwen_and_grok_drop_thinking_when_only_reasoning_comes_back(self):
        cases = (
            ("doubao-seed-2-0-lite", "thinking", {"type": "enabled"}, {"type": "disabled"}),
            ("qwen3.8-max", "enable_thinking", True, False),
            ("grok-4.7", "reasoning_effort", "high", "low"),
        )
        raw = {"choices":[{"finish_reason":"stop","message":{"content":"","reasoning":"private-chain"}}]}
        answer = {"choices":[{"finish_reason":"stop","message":{"content":"{\"reply\":\"成功\"}"}}]}
        for model, field, before, after in cases:
            with self.subTest(model=model):
                client = LLMClient({**self.options, "base_url": "https://gateway.example.test/v1", "model": model})
                bodies = [io.BytesIO(json.dumps(raw).encode()), io.BytesIO(json.dumps(answer).encode())]
                with patch("wxbot_framework.llm.urlopen", side_effect=bodies) as call:
                    self.assertEqual(client.complete_json("JSON", "测试"), {"reply": "成功"})
                first, second = [json.loads(args.args[0].data) for args in call.call_args_list]
                self.assertEqual(first[field], before)
                self.assertEqual(second[field], after)
                self.assertEqual(first["response_format"], {"type": "json_object"})
                self.assertEqual(second["response_format"], {"type": "json_object"})
                self.assertNotIn("private-chain", json.dumps(second, ensure_ascii=False))

    def test_refusal_filter_tool_call_and_truncation_do_not_retry(self):
        cases = [("content_filter",{}), ("stop",{"refusal":"private-provider-detail"}),
                 ("tool_calls",{}), ("stop",{"tool_calls":[{"id":"test"}]}), ("length",{})]
        for finish, fields in cases:
            raw = {"choices":[{"finish_reason":finish,"message":{"content":"",**fields}}]}
            with self.subTest(finish=finish,fields=fields), patch("wxbot_framework.llm.urlopen",return_value=io.BytesIO(json.dumps(raw).encode())) as call:
                with self.assertRaises(LLMError) as caught:
                    self.client.complete_json("JSON", "测试")
                self.assertEqual(call.call_count, 1)
                self.assertNotIn("private-provider-detail", str(caught.exception))

    def test_extension_cannot_override_protocol_or_authentication_fields(self):
        for key in ("model", "messages", "stream", "response_format", "max_tokens", "max_completion_tokens",
                    "tools", "web_search_options", "api_key", "Authorization"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "不能覆盖"):
                LLMClient({**self.options, "extra_body": {key: "forbidden"}})

    def test_invalid_extensions_and_compatibility_options(self):
        for change in ({"json_mode": "false"}, {"token_limit_parameter": "max_output_tokens"},
                       {"extra_body": []}, {"extra_body": None}, {"extra_body": {1: True}},
                       {"extra_body": {"temperature": float("nan")}},
                       {"extra_body": {"value": object()}}, {"extra_body": {"value": "a" * 16384}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LLMClient({**self.options, **change})

    def test_local_http_is_allowed_only_for_loopback_hosts(self):
        for url in ("http://localhost:11434/v1", "http://127.0.0.1:1234/v1", "http://[::1]:8080/v1"):
            with self.subTest(url=url):
                self.assertEqual(LLMClient({**self.options, "base_url": url}).endpoint, url + "/chat/completions")
        for url in ("http://192.168.1.2:8000/v1", "http://localhost.example.test/v1", "https://host:bad/v1",
                    "https://host:0/v1", "https://host/v1?token=private", "https://host/v1#fragment", "https://host/\n/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LLMClient({**self.options, "base_url": url})

    def test_http_errors_are_not_retried_or_returned_with_private_payloads(self):
        error = HTTPError("https://example.test", 400, "private prompt or credential", {}, None)
        with patch("wxbot_framework.llm.urlopen", side_effect=error) as call:
            with self.assertRaises(LLMError) as caught:
                self.client.complete_json("private prompt", "private content")
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("private", str(caught.exception))

    def test_config_from_utf8_bom_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm.json"
            path.write_text(json.dumps(self.options), encoding="utf-8-sig")
            self.assertEqual(LLMClient.from_file(path).model, "deepseek-flash")

    def test_invalid_configuration(self):
        for change in ({"api_key": ""}, {"model": 1}, {"base_url": None},
                       {"base_url": "http://example.com"}, {"base_url": "https://user:pass@example.com"},
                       {"timeout_seconds": True}, {"timeout_seconds": float("nan")},
                       {"max_tokens": 1}, {"max_tokens": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LLMClient({**self.options, **change})

    def test_server_and_connection_errors_hide_sensitive_details(self):
        for error in (HTTPError("https://example.com", 401, "dummy-test-key", {}, None),
                      URLError("dummy-test-key"), TimeoutError("dummy-test-key")):
            with self.subTest(error=type(error).__name__), patch("wxbot_framework.llm.urlopen", side_effect=error):
                with self.assertRaises(LLMError) as caught:
                    self.client.complete_json("JSON", "测试")
                self.assertNotIn("dummy-test-key", str(caught.exception))

    def test_malformed_or_truncated_model_results(self):
        for content in ("", "not json", "[]", None):
            with self.subTest(content=content), patch("wxbot_framework.llm.urlopen", return_value=self.response(content)):
                with self.assertRaises(LLMError):
                    self.client.complete_json("JSON", "测试")
        with patch("wxbot_framework.llm.urlopen", return_value=self.response("{}", finish_reason="length")):
            with self.assertRaisesRegex(LLMError, "截断"):
                self.client.complete_json("JSON", "测试")
        with patch("wxbot_framework.llm.urlopen", return_value=io.BytesIO(b'{"choices":[null]}')):
            with self.assertRaises(LLMError):
                self.client.complete_json("JSON", "测试")

    def _search(self, **overrides):
        settings = {"enabled": True, "provider": "http",
                    "base_url": "https://search.example.test/v1/tools/search",
                    "api_key": "search-key", "timeout_seconds": 8, "count": 2}
        settings.update(overrides)
        return LLMClient({**self.options, "web_search": settings})

    @staticmethod
    def _raw(item):
        return io.BytesIO(json.dumps(item).encode())

    @staticmethod
    def _calling(name, arguments, call_id="call_1", reasoning=None):
        message = {"content": "", "tool_calls": [{"id": call_id, "type": "function",
                                                  "function": {"name": name, "arguments": arguments}}]}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return {"choices": [{"finish_reason": "tool_calls", "message": message}]}

    def test_chat_without_tools_returns_plain_text(self):
        with patch("wxbot_framework.llm.urlopen", return_value=self.response(" 你好呀 ")) as call:
            self.assertEqual(self.client.chat("规则", "问题"), "你好呀")
        body = json.loads(call.call_args.args[0].data)
        self.assertNotIn("tools", body)
        self.assertNotIn("response_format", body)
        self.assertNotIn("JSON", body["messages"][0]["content"])

    def test_chat_runs_tools_and_passes_reasoning_back(self):
        seen = []
        tool = Tool("echo", "回声", {"type": "object", "properties": {}},
                    lambda args: seen.append(args) or "工具结果")
        bodies = [self._raw(self._calling("echo", '{"word":"嗨"}', reasoning="先查一下")),
                  self.response("最终回答")]
        with patch("wxbot_framework.llm.urlopen", side_effect=bodies) as call:
            self.assertEqual(self.client.chat("规则", "问题", [tool]), "最终回答")
        self.assertEqual(seen, [{"word": "嗨"}])
        first, second = [json.loads(item.args[0].data) for item in call.call_args_list]
        self.assertEqual(first["tools"][0]["function"]["name"], "echo")
        self.assertEqual(second["messages"][2]["reasoning_content"], "先查一下")
        self.assertEqual(second["messages"][2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(second["messages"][3], {"role": "tool", "tool_call_id": "call_1", "content": "工具结果"})
        self.assertNotIn("tool_choice", second)

    def test_tool_errors_go_back_to_the_model(self):
        def broken(args):
            raise LLMError("搜索服务挂了")
        tool = Tool("broken", "坏的", {"type": "object", "properties": {}}, broken)
        bodies = [self._raw(self._calling("broken", "{}")), self._raw(self._calling("missing", "{}", "call_2")),
                  self._raw(self._calling("broken", "not json", "call_3")), self.response("算了")]
        with patch("wxbot_framework.llm.urlopen", side_effect=bodies) as call:
            self.assertEqual(self.client.chat("规则", "问题", [tool]), "算了")
        messages = json.loads(call.call_args_list[-1].args[0].data)["messages"]
        results = [item["content"] for item in messages if item["role"] == "tool"]
        self.assertEqual(results, ["工具出错：搜索服务挂了", "没有这个工具", "工具参数应为 JSON 对象"])

    def test_tool_rounds_are_bounded(self):
        tool = Tool("echo", "回声", {"type": "object", "properties": {}}, lambda args: "x")
        bodies = [self._raw(self._calling("echo", "{}", f"call_{index}")) for index in range(MAX_TOOL_ROUNDS)]
        with patch("wxbot_framework.llm.urlopen", side_effect=bodies + [self.response("收尾")]) as call:
            self.assertEqual(self.client.chat("规则", "问题", [tool]), "收尾")
        self.assertEqual(call.call_count, MAX_TOOL_ROUNDS + 1)
        self.assertEqual(json.loads(call.call_args_list[-1].args[0].data)["tool_choice"], "none")
        self.assertNotIn("tool_choice", json.loads(call.call_args_list[-2].args[0].data))

    def test_model_without_tool_support_retries_without_tools(self):
        tool = Tool("echo", "回声", {"type": "object", "properties": {}}, lambda args: "x")
        error = HTTPError("https://example.test", 400, "tools unsupported", {}, None)
        with patch("wxbot_framework.llm.urlopen", side_effect=[error, self.response("好")]) as call:
            self.assertEqual(self.client.chat("规则", "问题", [tool]), "好")
        first, second = [json.loads(item.args[0].data) for item in call.call_args_list]
        self.assertIn("tools", first)
        self.assertNotIn("tools", second)

    def test_blank_chat_reply_retries_once_then_fails(self):
        raw = {"choices": [{"finish_reason": "stop", "message": {"content": "", "reasoning_content": "想想"}}]}
        with patch("wxbot_framework.llm.urlopen", side_effect=[self._raw(raw), self.response("想好了")]) as call:
            self.assertEqual(self.client.chat("规则", "问题"), "想好了")
        self.assertEqual(json.loads(call.call_args_list[1].args[0].data)["thinking"], {"type": "disabled"})
        with patch("wxbot_framework.llm.urlopen", side_effect=[self._raw(raw), self._raw(raw)]):
            with self.assertRaisesRegex(LLMError, "空结果"):
                self.client.chat("规则", "问题")

    def test_closed_search_is_not_offered_as_a_tool(self):
        names = [tool.name for tool in chat_tools(self.client, None, "room")]
        self.assertNotIn("web_search", names)
        self.assertIn("web_search", [tool.name for tool in chat_tools(self._search(), None, "room")])

    def test_http_search_tool_uses_its_own_key_or_the_model_key(self):
        found = {"results": [{"title": "标题", "url": "https://example.com/a", "summary": "摘要"},
                             {"title": "天气", "url": "javascript:x", "snippet": "晴"}]}
        for key, expected in (("search-key", "Bearer search-key"), ("", "Bearer dummy-test-key")):
            with self.subTest(key=key), patch("wxbot_framework.llm.urlopen", return_value=self._raw(found)) as call:
                result = json.loads(web_search_tool(self._search(api_key=key)).run({"query": "今天新闻"}))
            request = call.call_args.args[0]
            self.assertEqual(request.full_url, "https://search.example.test/v1/tools/search")
            self.assertEqual(request.get_header("Authorization"), expected)
            self.assertEqual(json.loads(request.data), {"query": "今天新闻", "count": 2, "need_content": False})
            self.assertEqual(result, [{"title": "标题", "url": "https://example.com/a", "summary": "摘要"},
                                      {"title": "天气", "url": "", "summary": "晴"}])

    def test_search_transport_failure_reaches_the_model_as_a_tool_error(self):
        with patch("wxbot_framework.llm.urlopen", side_effect=URLError("down")):
            with self.assertRaisesRegex(LLMError, "联网搜索连接失败"):
                web_search_tool(self._search()).run({"query": "新闻"})
        with self.assertRaisesRegex(ValueError, "query"):
            web_search_tool(self._search()).run({"query": "  "})

    def test_doubao_search_tool_is_a_separate_request_without_tools(self):
        client = LLMClient({**self.options, "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                            "model": "doubao-seed-2-0-lite",
                            "web_search": {"enabled": True, "provider": "doubao"}})
        answer = {"choices": [{"message": {"content": "今天晴", "annotations": [
            {"type": "url_citation", "url_citation": {"url": "https://weather.example/a"}}]}}]}
        with patch("wxbot_framework.llm.urlopen", return_value=self._raw(answer)) as call:
            self.assertEqual(web_search_tool(client).run({"query": "天气"}), "今天晴\n来源：https://weather.example/a")
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(call.call_args.args[0].full_url, "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
        self.assertEqual(body["web_search_options"], {})
        self.assertNotIn("tools", body)
        self.assertEqual(body["messages"][1]["content"], "天气")

    def test_vision_client_falls_back_to_the_dialogue_model(self):
        self.assertIs(self.client.vision_client(), self.client)
        client = LLMClient({**self.options, "extra_body": {"temperature": 0.5},
                            "vision": {"model": "qwen-vl", "base_url": "https://vision.example.test/v1"}})
        vision = client.vision_client()
        self.assertEqual((vision.model, vision.base_url, vision.api_key, vision.extra_body),
                         ("qwen-vl", "https://vision.example.test/v1", "dummy-test-key", {}))
        with self.assertRaises(ValueError):
            LLMClient({**self.options, "vision": {"base_url": "http://example.com"}})

    def test_http_search_requires_a_url_when_enabled(self):
        with self.assertRaisesRegex(ValueError, "base_url"):
            self._search(base_url="", api_key="")


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.options = {"enabled": True, "api_key": "decision-key", "fallback": {"speak": 0.2, "voice": 0.7}}

    @staticmethod
    def answer(value):
        return io.BytesIO(json.dumps({"answers": {"speak": {"noul": value}}}).encode())

    def test_model_answer_is_compared_with_the_threshold(self):
        decider = Decider(self.options)
        with patch("wxbot_framework.llm.urlopen", return_value=self.answer(0.6)) as call:
            self.assertTrue(decider.yes("speak", "该说话吗", ("该说", "不该说"), {"chat": "大家好"}))
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://emrank.tech/v1/decisions")
        self.assertEqual(request.get_header("Authorization"), "Bearer decision-key")
        self.assertEqual(request.get_header("User-agent"), "helper/1.0")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "typesafe/jev[slow]")
        self.assertEqual(body["state"], {"chat": "大家好"})
        self.assertEqual(body["questions"], {"speak": {"type": "noul", "instructions": "该说话吗",
                                                       "criteria": {"true": "该说", "false": "不该说"}}})
        with patch("wxbot_framework.llm.urlopen", return_value=self.answer(0.6)):
            self.assertFalse(decider.yes("speak", "该说话吗", ("该说", "不该说"), {}, threshold=0.7))

    def test_configured_url_is_posted_as_is(self):
        url = "https://llm.example.test/compatible-mode/v1/systemone"
        decider = Decider({**self.options, "base_url": url + "/"})
        with patch("wxbot_framework.llm.urlopen", return_value=self.answer(0.4)) as call:
            self.assertFalse(decider.yes("speak", "问", ("是", "否"), {"ticket": "T-1"}))
        self.assertEqual(call.call_args.args[0].full_url, url)
        self.assertEqual(normalize_decision({"base_url": "https://emrank.tech/v1"})["base_url"],
                         "https://emrank.tech/v1/decisions")

    def test_long_state_keeps_only_the_tail(self):
        with patch("wxbot_framework.llm.urlopen", return_value=self.answer(0.9)) as call:
            Decider(self.options).yes("speak", "问", ("是", "否"), {"chat": "旧" * 20000 + "最新"})
        context = json.loads(call.call_args.args[0].data)["state"]["context"]
        self.assertTrue(context.endswith('最新"}'))
        self.assertLessEqual(len(context), 12000)

    def test_unavailable_model_falls_back_to_the_configured_chance(self):
        failures = (URLError("down"), HTTPError("https://emrank.tech", 403, "blocked", {}, None),
                    self.answer(1.5), self.answer("0.9"), io.BytesIO(b'{"answers":{}}'))
        for failure in failures:
            with self.subTest(failure=type(failure).__name__), patch(
                    "wxbot_framework.llm.urlopen", side_effect=[failure]), patch(
                    "wxbot_framework.decision.random.random", return_value=0.1):
                self.assertTrue(Decider(self.options).yes("speak", "问", ("是", "否"), {}))
        with patch("wxbot_framework.llm.urlopen") as call, patch(
                "wxbot_framework.decision.random.random", return_value=0.5):
            self.assertFalse(Decider({**self.options, "enabled": False}).yes("speak", "问", ("是", "否"), {}))
            self.assertTrue(Decider({**self.options, "enabled": False}).yes("voice", "问", ("是", "否"), {}))
        call.assert_not_called()

    def test_settings_validate_and_hide_the_key(self):
        self.assertEqual(normalize_decision(None)["fallback"], {"speak": 0.3, "voice": 0.3})
        for change in ({"api_key": ""}, {"base_url": "http://example.com"}, {"timeout_seconds": 0},
                       {"fallback": {"speak": 2}}, {"fallback": {"voice": True}}, {"enabled": "yes"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_decision({**self.options, **change})
        shown = public_decision(self.options)
        self.assertNotIn("api_key", shown)
        self.assertTrue(shown["api_key_configured"])
        with self.assertRaises(ValueError):
            Decider(None).yes("other", "问", ("是", "否"), {})


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.client = JSONClient({"is_idiom": True, "reason": "常见成语", "next_idiom": "发扬光大"})
        self.judge = IdiomJudge(self.client)

    def test_local_rejections_do_not_call_model(self):
        for previous, word, used in (("一心一意", "不是四字成语", ()),
                                     ("一心一意", "异想天开", ()),
                                     ("一心一意", "意气风发", ("意气风发",))):
            self.assertFalse(self.judge.evaluate(previous, word, used).accepted)
        self.assertEqual(self.client.calls, [])

    def test_success_sends_only_game_words(self):
        result = self.judge.evaluate("一心一意", "意气风发", ("一心一意",))
        self.assertTrue(result.accepted)
        self.assertEqual(result.next_idiom, "发扬光大")
        self.assertEqual(json.loads(self.client.calls[0][1]),
                         {"previous": "一心一意", "candidate": "意气风发", "used": ["一心一意", "意气风发"]})

    def test_model_can_reject_fake_four_character_word(self):
        self.client.result = {"is_idiom": False, "reason": "不是成语", "next_idiom": ""}
        result = self.judge.evaluate("一心一意", "意意意意", ("一心一意",))
        self.assertFalse(result.accepted)

    def test_invalid_model_next_word_never_advances(self):
        for word in ("假的", "一心一意", "发扬光大", "意气风发"):
            self.client.result["next_idiom"] = word
            with self.subTest(word=word), self.assertRaises(LLMError):
                self.judge.evaluate("一心一意", "意气风发", ("一心一意", "发扬光大"))

    def test_model_boolean_is_strict(self):
        self.client.result["is_idiom"] = "true"
        with self.assertRaises(LLMError):
            self.judge.evaluate("一心一意", "意气风发", ())

    def test_opening_shape(self):
        self.client.result = {"idiom": "一心一意"}
        self.assertEqual(self.judge.opening(), "一心一意")
        self.client.result = {"idiom": "假词"}
        with self.assertRaises(LLMError):
            self.judge.opening()


class VisionRequestTests(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient({"api_key": "dummy-test-key", "model": "deepseek-v4-flash", "timeout_seconds": 20})

    def response(self, content):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    def test_image_is_a_data_url_and_uses_a_longer_timeout(self):
        image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
        with patch("wxbot_framework.llm.urlopen", return_value=self.response('{"reply":"红色"}')) as call:
            self.assertEqual(self.client.complete_json("只返回 JSON", "看这张图", image=image), {"reply": "红色"})
        self.assertGreaterEqual(call.call_args.kwargs["timeout"], 45)
        payload = json.loads(call.call_args.args[0].data)
        self.assertTrue(payload["messages"][0]["content"].startswith("只返回 JSON"))
        self.assertIn("JSON 对象", payload["messages"][0]["content"])
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        parts = payload["messages"][1]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "看这张图"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["url"], image)
        self.assertNotIn(image, str(call.call_args.args[0].headers))

    def test_remote_or_unknown_image_is_rejected_before_the_request(self):
        for image in ("https://example.test/a.png", "data:image/bmp;base64,QQ==", "data:image/png;base64,"):
            with self.subTest(image=image[:24]), patch("wxbot_framework.llm.urlopen") as call:
                with self.assertRaises(LLMError) as caught:
                    self.client.complete_json("JSON", "看图", image=image)
                self.assertIsNone(caught.exception.status)
                call.assert_not_called()

    def test_http_400_keeps_status_and_hides_the_image(self):
        image = "data:image/jpeg;base64," + ("A" * 40)
        error = HTTPError("https://example.test", 400, image, {}, None)
        with patch("wxbot_framework.llm.urlopen", side_effect=error):
            with self.assertRaises(LLMError) as caught:
                self.client.complete_json("JSON", "看图", image=image)
        self.assertEqual(caught.exception.status, 400)
        self.assertNotIn("AAAA", str(caught.exception))


class SeenImageTests(unittest.TestCase):
    def test_recent_image_expires_but_quoted_id_remains_until_evicted(self):
        images = SeenImages(window=180, limit=1)
        self.assertFalse(images.remember("room", "amy", 0, "zero.png"))
        self.assertFalse(images.remember("room", "amy", True, "bool.png"))
        self.assertTrue(images.remember("room", "amy", "11", "first.png"))
        self.assertEqual(images.latest("room", "amy"), "first.png")
        self.assertEqual(images.latest("room", "other"), "")
        self.assertEqual(images.latest("room", "amy", now=time.monotonic() + 181), "")
        self.assertEqual(images.by_id("11"), "first.png")
        self.assertTrue(images.remember("room", "amy", "22", "second.png"))
        self.assertEqual(images.by_id("11"), "")
        self.assertEqual(images.by_id(22), "second.png")

    def test_local_image_becomes_a_supported_data_url(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            small = root / "small.png"
            wide = root / "wide.png"
            bitmap = root / "plain.bmp"
            Image.new("RGB", (8, 8), "red").save(small)
            Image.new("RGB", (1400, 8), "blue").save(wide)
            Image.new("RGB", (8, 8), "green").save(bitmap)
            (root / "note.txt").write_bytes(b"not an image")
            png = vision_data_url(small)
            jpeg = vision_data_url(wide)
            converted = vision_data_url(bitmap)
            self.assertTrue(png.startswith("data:image/png;base64,"))
            self.assertTrue(jpeg.startswith("data:image/jpeg;base64,"))
            self.assertTrue(converted.startswith("data:image/jpeg;base64,"))
            self.assertNotEqual(png, jpeg)
            self.assertEqual(vision_data_url(root / "note.txt"), "")
