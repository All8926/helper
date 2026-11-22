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
    """只组装这个人。这次怎么接话由全局提示词决定。"""
    layers = layers if isinstance(layers, dict) else {}
    before = _filled(layers, _BEFORE_VOICE)
    after = _filled(layers, _VOICE)
    body = [f"你就是{name}。下面是你本人。"]
    if before:
        body.append(before)
    if after:
        body.append(after)
    return "\n\n".join(body)


def system_with_personality(system, state, prompts=None, *, plain=False):
    """有人格时，人在前，全局提示词决定这次接谁、查不查、写多长。"""
    raw = system.strip() if isinstance(system, str) else ""
    marker = "【参考背景"
    if marker in raw:
        idx = raw.index(marker)
        task, note = raw[:idx].strip(), raw[idx:].strip()
    else:
        task, note = raw, ""
    tool_rule = (
        "你在微信里直接回话：assistant 消息是你之前亲口说过的原话，user 消息是群友或对方的发言。"
        "你只输出这次要发出去的回复正文，不加自己的名字前缀、不替别人发言、也不原样复读对方的话。\n"
        "工具在本轮回复发出前同步执行：消息里标有 [图片 id=xxx] 或 [引用图片 id=xxx] 时，"
        "只要回应需要知道图里内容，就立刻发起 view_image 工具调用（传入对应 id），拿到画面描述后再开口；"
        "一旦不调工具直接输出文字，本轮对话就会立刻发出并结束，所以绝不要用文字回复“我先看看图”“等我查一下”。"
        "不需要看图或查资料时直接回复，没看过图就别装看见。"
    )
    if not plain or state.name == "默认":
        parts = [task or "直接回应最后一条用户发言。"]
        if note:
            parts.append(note)
        parts.append(tool_rule)
        return "\n\n".join(parts)
    layers = ({} if prompts is None else prompts).get(state.name)
    if not isinstance(layers, dict):
        parts = [task] if task else []
        if note:
            parts.append(note)
        parts.append(tool_rule)
        return "\n\n".join(parts)
    parts = [compose_character(state.name, layers),
             f"你是{state.name}，说话方式照上面。下面只决定这次怎么接话与调用工具，不能把你换成别人。",
             task or "直接回应最后一条用户发言。"]
    if note:
        parts.append(note)
    parts.append(tool_rule)
    return "\n\n".join(parts)


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
