"""点名对话。模型看最近一小段流水、近两天的日记和当前说话人的档案，其余资料靠工具自己查。"""
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


def checked_reply(text, limit):
    """模型正文的最后检查：不能为空、不能过长、不能带控制字符。"""
    if (not isinstance(text, str) or not text.strip() or len(text) > limit or
            any(ord(char) < 32 and char not in "\n\r\t" for char in text)):
        raise LLMError("大模型对话结果为空或过长，请重试")
    return text.strip()


class Dialogue:
    def __init__(self, client):
        self.client = client

    def reply(self, snapshot, *, diary=(), memory=(), tools=(), system="", limit=600):
        """返回要发到群里的正文。system 是本账号的全局提示词，空则只有人格包装。"""
        transcript = [transcript_line(row, current=False) for row in working_messages(snapshot.messages)]
        transcript.append({"time": datetime.now(BEIJING).strftime("%H:%M:%S"),
                           "speaker": snapshot.speaker_name, "type": "text",
                           "text": transcript_text(strip_at_labels(snapshot.current)), "current": True})
        body = {"name": snapshot.speaker_name, "date": snapshot.day, "diary": public_diary(diary),
                "memory": public_memory(memory), "transcript": transcript}
        return checked_reply(self.client.chat(system or "", json.dumps(body, ensure_ascii=False), tools), limit)
