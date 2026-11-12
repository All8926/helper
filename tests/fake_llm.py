"""测试用的大模型客户端：不发请求，chat 和 complete_json 交给传入的函数，并记下每次调用。"""
from unittest.mock import Mock

from wxbot_framework.decision import normalize_decision
from wxbot_framework.llm import normalize_vision, normalize_web_search


class FakeLLM:
    def __init__(self, chat=None, complete_json=None, *, decision=None, web_search=None):
        self.chat = Mock(side_effect=chat or (lambda system, user, tools=(), **_kwargs: "这是回答。"))
        self.complete_json = Mock(side_effect=complete_json or (lambda system, user, **_kwargs: {}))
        self.web_search = normalize_web_search(web_search)
        self.vision = normalize_vision(None)
        self.decision = normalize_decision(decision)
        self.api_key, self.base_url, self.model = "dummy-test-key", "https://api.deepseek.com", "deepseek-flash"

    def vision_client(self):
        return self
