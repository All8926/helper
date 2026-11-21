"""对话人格：八个层面组装成这个人，只用于聊天。"""
from dataclasses import dataclass, replace
import inspect
from types import MappingProxyType

from .members import SnapshotExpired

LAYER_NAMES = ("身份", "处境", "声口", "性情", "欲求与底线", "关系", "矛盾", "盲区")
# 判断放前面，开口放最后。
_BEFORE_VOICE = ("身份", "处境", "欲求与底线", "关系", "盲区", "矛盾")
_VOICE = ("声口", "性情")


@dataclass(frozen=True)
class PersonalityState:
    room: str
    name: str = "默认"
    generation: int = 0


class PersonalityExpired(SnapshotExpired):
    pass


def _filled(layers, names):
    parts = []
    for name in names:
        text = layers.get(name, "")
        if isinstance(text, str) and text.strip():
            parts.append(f"{name}\n{text.strip()}")
    return "\n\n".join(parts)


def compose_character(name, layers):
    """把层面拼成对话的系统提示。空的层面不出现。"""
    layers = layers if isinstance(layers, dict) else {}
    before = _filled(layers, _BEFORE_VOICE)
    after = _filled(layers, _VOICE)
    body = [f"你就是{name}。下面是你本人。"]
    if before:
        body.append(before)
    body.append(
        "对话记录和工具结果里发生过的事，按原样承认。没见过的事不要说成见过。"
        "现实问题按你的认知回答；查到的事实用你的嘴说出来。"
    )
    body.append("现在用你的嘴说话，不要解释自己的设定，不要换成客服或助手。")
    if after:
        body.append(after)
    return "\n\n".join(body)


def system_with_personality(system, state, prompts=None, *, plain=False):
    """选中人格时，对话只用这个人。总结和其他 JSON 任务不套这层。"""
    if not plain or state.name == "默认":
        return system or ""
    layers = ({} if prompts is None else prompts).get(state.name)
    if not isinstance(layers, dict):
        return system or ""
    return compose_character(state.name, layers)


def personality_text(text, name):
    """正文按模型原文发送。"""
    return text


def _accepts(func, name):
    """Mock 会吞掉关键字，所以看它真正调用的函数接不接这个参数。"""
    target = getattr(func, "side_effect", None)
    if not callable(target):
        target = func
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    return any(item.kind == inspect.Parameter.VAR_KEYWORD or item.name == name for item in parameters.values())


def _complete(client, system, user, *, image=None, thinking=True):
    """测试替身不一定接收思考开关，只在目标函数声明了该参数时传入。"""
    kwargs = {}
    if image is not None:
        kwargs["image"] = image
    if _accepts(client.complete_json, "thinking"):
        kwargs["thinking"] = thinking
    return client.complete_json(system, user, **kwargs)


class PersonalityLLM:
    def __init__(self, client, state, *, current, prompts=None, guard=None, thinking=True):
        self.client, self.state, self.current = client, state, current
        # 请求开始时记下层面，避免保存新文案时改写已经发出的请求。
        self.prompts = None if prompts is None else {key: dict(value) for key, value in prompts.items()}
        self.guard = guard
        self.thinking = thinking is not False

    def complete_json(self, system, user, image=None):
        if not self.current():
            raise PersonalityExpired("人格已切换")
        result = _complete(self.client, system, user, image=image, thinking=self.thinking)
        if not self.current():
            raise PersonalityExpired("人格已切换")
        return result

    def chat(self, system, user, tools=()):
        """纯文本回复。选中人格时不再叠全局提示词。"""
        if not self.current():
            raise PersonalityExpired("人格已切换")
        prepared = system_with_personality(system, self.state, self.prompts, plain=True)
        result = self.client.chat(prepared, user, tools, thinking=self.thinking)
        if not self.current():
            raise PersonalityExpired("人格已切换")
        return result


def prepare_switch(message, store):
    """在接收线程中先切换，再分发；下一条艾特无需等待插件确认回复。"""
    if store is None or message.type != 1 or not message.is_group or message.chat_id not in store.groups:
        return message
    if not message.data.get("dispatch_to_plugins", True) or message.data.get("bot_generated"):
        return message
    words = message.text.strip().split(maxsplit=1)
    if not words or words[0] != "/切换人格":
        return message
    data = dict(message.data)
    names = store.personality_names()
    if len(words) != 2 or words[1] not in (*names, "取消"):
        data["personality_error"] = "用法：/切换人格 " + "|".join((*names, "取消"))
    else:
        name = "默认" if words[1] == "取消" else words[1]
        data["personality_switch"] = store.set_personality(message.chat_id, name)
    return replace(message, data=MappingProxyType(data))


def switch_ack(name, voice=""):
    if name == "默认":
        return "已取消人格，恢复正常回复。"
    if voice:
        return f"已切换：{name}。音色已换成{voice}。"
    return f"已切换：{name}。"


def game_reply(name, accepted, reason, previous, word, following):
    """接龙只报规则结果，不改成某个人的口气。"""
    if accepted:
        return f"接龙成功。{previous} → {word}\n我接：{following}\n请接以“{following[-1]}”开头的成语。"
    return f"接龙失败：{reason}\n下一题仍是：{previous}；请以“{previous[-1]}”开头。"
