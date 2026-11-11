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

from wxbot_framework.llm import LLMClient, LLMError
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
        for key in ("model", "messages", "stream", "response_format", "max_tokens", "max_completion_tokens", "api_key", "Authorization"):
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
