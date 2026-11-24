"""点名对话。聊天记录拆成真实 user/assistant 轮次，近两天日记和个人档案放入系统背景，其余资料靠工具自己查。"""
import json
import re
from datetime import datetime

from .llm import LLMError
from .members import BEIJING
from .memory import (LINE_LIMIT, PROMPT_CHARS, PROMPT_LINES, public_diary, public_memory)
from .mentions import strip_at_labels

PROMPT_DIARY_DAYS = 2

VOICE_QUESTION = "群聊机器人的这句回复适合念成语音发出去吗？"
VOICE_CRITERIA = (
    "结合当前这句话和最近的聊天，这句回复适合直接听",
    "回复里有链接、代码、长串数字、列表等需要看的文字，或者结合当前这句话和最近的聊天，念出来并不合适",
)
VOICE_CONTEXT_LINES = 8
VOICE_REPLY_LIMIT = 300
SING_QUESTION = "这条回复正文是不是歌词？"
SING_CRITERIA = (
    "整段回复就是歌词，不是解释、问答或普通聊天",
    "这是白话回复、解释、问答或普通聊天，不是歌词",
)
_LYRIC_BLOCK = re.compile(r"https?://|www\.\S+|```")


def sing_state(reply):
    """清唱决策只看这条回复，不附聊天，也不按字数截断。"""
    return {"reply": str(reply or "")}


def singable(text):
    """链接、代码、长串数字或列表不送去清唱。字数不在这里卡。"""
    raw = str(text or "")
    collapsed = " ".join(raw.split())
    if not collapsed or _LYRIC_BLOCK.search(raw):
        return False
    if any(line.lstrip().startswith(("- ", "* ", "1.", "1、")) for line in raw.splitlines()):
        return False
    digits = sum(ch.isdigit() for ch in collapsed)
    return not (digits >= 6 and digits * 2 >= len(collapsed))


def _voice_line(row, bot_id):
    """语音决策用的一条聊天。只去掉换行和 @ 标签，不截断正文。"""
    text = " ".join(strip_at_labels(str(row.get("text") or "")).split())
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(row["received_at"]).astimezone(BEIJING).strftime("%H:%M:%S")
    except (TypeError, ValueError, KeyError):
        moment = ""
    bot = bool(row.get("bot") or row.get("bot_generated") or (bot_id and row.get("sender_id") == bot_id))
    return {"time": moment, "speaker": row.get("speaker") or "群友", "text": text, "bot": bot}


def voice_state(heard, reply, rows, bot_id):
    """送给决策模型的语音题材料。回复可以截断，当前这句话和最近几条聊天保持原文。"""
    message = " ".join(strip_at_labels(str(heard or "")).split())
    lines = [line for row in rows if (line := _voice_line(row, bot_id))]
    return {"message": message, "reply": str(reply or "")[:VOICE_REPLY_LIMIT],
            "transcript": lines[-VOICE_CONTEXT_LINES:]}


def transcript_line(row, *, current):
    """每条消息带类型。图片只给还能打开的编号，不给本地路径。"""
    moment = datetime.fromisoformat(row["received_at"]).astimezone(BEIJING).strftime("%H:%M:%S")
    line = {"time": moment, "speaker": row.get("speaker") or "群友", "current": current}
    if row.get("kind") == "image":
        line["type"] = "image"
        token = row.get("media_ref") or ""
        if token:
            line["id"] = token
        return line
    line["type"] = "text"
    line["text"] = transcript_text(strip_at_labels(row.get("text") or ""))
    return line


def transcript_text(text):
    """单条流水压成一行，过长时截断，避免一条长消息占满上下文。"""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= LINE_LIMIT:
        return collapsed
    return collapsed[:LINE_LIMIT] + "..."


def working_messages(messages):
    """读取窗口可能有几百条，送给模型的只留最新一小段。"""
    chosen = []
    size = 0
    for row in reversed(messages):
        text = transcript_text(strip_at_labels(row.get("text") or "")) or ("图" if row.get("kind") == "image" else "")
        if len(chosen) >= PROMPT_LINES:
            break
        if chosen and size + len(text) > PROMPT_CHARS:
            break
        chosen.append(row)
        size += len(text)
    chosen.reverse()
    return chosen


def _moment_hhmm(received_at):
    try:
        return datetime.fromisoformat(str(received_at)).astimezone(BEIJING).strftime("%H:%M")
    except (TypeError, ValueError):
        return ""


def _human_line(row, *, private=False):
    if row.get("kind") == "image":
        # 图片只进参考背景。写进对话正文会被模型原样说出去。
        return ""
    speaker = row.get("speaker") or ("对方" if private else "群友")
    moment = _moment_hhmm(row.get("received_at"))
    prefix = f"[{moment}] " if moment else ""
    text = transcript_text(strip_at_labels(row.get("text") or ""))
    if not text:
        return ""
    return f"{prefix}{text}" if private else f"{prefix}{speaker}：{text}"


def _append_turn(turns, role, line):
    if not line:
        return
    if turns and turns[-1]["role"] == role:
        turns[-1] = {"role": role, "content": turns[-1]["content"] + "\n" + line}
    else:
        turns.append({"role": role, "content": line})


def history_turns(rows, *, private=False, bot_id=""):
    """把历史记录拆成真实的 user/assistant 对话轮次，连续同角色合并。"""
    turns = []
    for row in rows:
        is_bot = bool(row.get("bot") or row.get("bot_generated") or (bot_id and row.get("sender_id") == bot_id))
        if is_bot:
            text = transcript_text(strip_at_labels(row.get("text") or ""))
            _append_turn(turns, "assistant", text)
            continue
        _append_turn(turns, "user", _human_line(row, private=private))
    return turns


def image_catalog(rows):
    """图片编号只放进参考背景，对话正文里不出现这串内部记号。"""
    items, seen = [], set()
    for row in rows:
        if row.get("kind") != "image":
            continue
        token = row.get("media_ref") or ""
        if not isinstance(token, str) or not token or token in seen:
            continue
        seen.add(token)
        speaker = row.get("speaker") or "群友"
        moment = _moment_hhmm(row.get("received_at"))
        when = f"{moment} " if moment else ""
        items.append(f"{token}（{when}{speaker}发的）")
    return items


def context_note(day, *, speaker="", diary=(), memory=(), people=(), images=(), unnamed=False,
                 focus_image=False):
    """把日期、日记、成员档案与当前会话对象整理成系统背景，不塞进用户发言。"""
    lines = [f"当前日期：{day}"]
    if unnamed:
        lines.append("当前情境：群里自然接话（未点名），直接对群里接最后一句，不艾特、不引用。")
    elif speaker:
        lines.append(f"当前对话人：{speaker}")
    cleaned_diary = public_diary(diary)
    if cleaned_diary:
        lines.append("近期群日记：" + "；".join(f"{item['date']} {item['summary']}" for item in cleaned_diary))
    cleaned_memory = public_memory(memory)
    if cleaned_memory:
        target = speaker or "对方"
        lines.append(f"关于{target}的已知信息：" + "；".join(cleaned_memory))
    for item in people:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or "群友"
        facts = public_memory(item.get("memory") or ())
        if facts:
            lines.append(f"关于{name}的已知信息：" + "；".join(facts))
    if images:
        lines.append("可查看图片（编号只传给 view_image，不要写进回复）：" + "；".join(images))
        if focus_image:
            lines.append("此刻要接的是可查看图片里的最后一张，直接说画面里的内容。")
    return "【参考背景（仅作已知背景，不是新提问）】\n" + "\n".join(lines)


def merge_system(system, note):
    base = (system or "").strip()
    extra = (note or "").strip()
    if base and extra:
        return base + "\n\n" + extra
    return base or extra


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def _plain_reply(text):
    """拆掉 action/text 外壳。解释 current、不需要回应或输出空的，当成没写。"""
    raw = text.strip()
    fenced = _FENCE.fullmatch(raw)
    if fenced:
        raw = fenced.group(1).strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and "action" in data:
            body = data.get("text", data.get("reply"))
            raw = body.strip() if isinstance(body, str) else ""
    if re.search(r"current", raw, re.I) and re.search(r"为\s*(?:true|false)|不需要回应|输出空", raw, re.I):
        return ""
    return raw


def checked_reply(text, limit):
    """取出可发送的话。空的和带控制字符的失败；超长则截到上限再发。"""
    if not isinstance(text, str):
        raise LLMError("大模型对话结果为空，请重试")
    reply = _plain_reply(text)
    if not reply or any(ord(char) < 32 and char not in "\n\r\t" for char in reply):
        raise LLMError("大模型对话结果为空，请重试")
    if len(reply) > limit:
        reply = reply[:limit].rstrip()
    if not reply:
        raise LLMError("大模型对话结果为空，请重试")
    return reply


def quoted_image_token(message, images):
    """引用图的编号。只放进参考背景，不写进对话正文。"""
    from .quoted import reference
    quoted = reference(message)
    if quoted is None or quoted.kind != "image":
        return ""
    native = message.data.get("quoted_message_id")
    token = images.token_for(native) if native else ""
    return token if isinstance(token, str) else ""


def quote_hint(message):
    """提取当前发言引用的图片或文字摘要，直接附在当前发言前。编号不放这里。"""
    from .quoted import reference
    quoted = reference(message)
    if quoted is None:
        return ""
    if quoted.kind == "image":
        return ""
    summary = transcript_text(quoted.text or "")
    if not summary:
        return ""
    author = f"{quoted.author}：" if quoted.author else ""
    return f"[引用 {author}{summary}]"


class Dialogue:
    def __init__(self, client):
        self.client = client

    def reply(self, snapshot, *, diary=(), memory=(), tools=(), system="", limit=600, quote="",
              quoted_image=""):
        """返回要发出去的正文。system 是本账号的全局提示词，背景与历史拆为真实多轮对话。"""
        private = not str(snapshot.room).endswith("@chatroom")
        rows = working_messages(snapshot.messages)
        turns = history_turns(rows, private=private)
        moment = datetime.now(BEIJING).strftime("%H:%M")
        current = transcript_text(strip_at_labels(snapshot.current))
        speaker = snapshot.speaker_name or ("对方" if private else "群友")
        prefix = f"{quote} " if quote else ""
        if current:
            last_line = f"[{moment}] {prefix}{current}" if private else f"[{moment}] {speaker}：{prefix}{current}"
        else:
            call = "（叫了你一声）" if private else "（艾特了你）"
            last_line = f"[{moment}] {prefix}{call}" if private else f"[{moment}] {speaker}：{prefix}{call}"
        turns.append({"role": "user", "content": last_line})
        images = image_catalog(rows)
        if quoted_image and all(not item.startswith(f"{quoted_image}（") for item in images):
            images.append(f"{quoted_image}（当前引用的）")
        note = context_note(snapshot.day, speaker=speaker, diary=diary, memory=memory, images=images)
        return checked_reply(self.client.chat(merge_system(system, note), turns, tools), limit)
