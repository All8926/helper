"""未点名回应：没人点名时，机器人什么时候问决策模型“现在会不会自然接一句”。

有人说完，等群里安静几秒再判断。平时也按看群间隔看一眼。
两种时机说出去的话，都算进本群每小时上限。点名回复不算。"""
from __future__ import annotations

import json
import threading
import time

QUIET_SECONDS = 6
HOUR = 3600
RETRY_SECONDS = 180

SPEAK_QUESTION = "机器人已经是这个群里的一员。看着标成 current 的最新一条，现在会不会自然接一句？"
SPEAK_CRITERIA = (
    "最新一条是抛给大家的：观点、玩笑、吐槽、经历、问句，或在接别人刚才的话。群还敞着，这时接一句就像群里的人",
    "最新一条只是嗯、哈哈、纯表情或斗图，没有可接的内容；或者两个人明显在说只有他们之间的事；或者机器人上一条还挂着、没人接；或者再开口就是重复、抢话",
)


def is_bot(row, bot_id):
    return bool(row.get("bot") or (bot_id and row.get("sender_id") == bot_id))


def clean_reply(value):
    """压成一行，去掉首尾空白。"""
    return " ".join(str(value or "").split())


def unnamed_input(day, transcript, diary, people):
    return json.dumps({"date": day, "diary": diary, "people": people, "transcript": transcript},
                      ensure_ascii=False)


class RoomListener:
    """每个群记：机器人上次说话的时间、上次看过的人类发言、
    最近一小时主动说出去的时间点，以及等人说完再判断的截止时间。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._rooms = {}

    def _state(self, room):
        return self._rooms.setdefault(room, {"spoke": None, "seen": None, "glanced": None, "beats": [],
                                             "due": None, "retry_after": None})

    def bot_spoke(self, room, *, beat=False, now=None):
        """机器人说了一句。beat 为真时，这句算进每小时主动说话上限。"""
        moment = time.monotonic() if now is None else now
        with self._lock:
            state = self._state(room)
            state["spoke"] = moment
            if beat:
                state["beats"].append(moment)

    def addressed(self, room):
        """有人点名，交给点名回复，丢掉还没判断的那次。"""
        with self._lock:
            self._state(room)["due"] = None

    def note_failed(self, room, now=None):
        """这次没发出去。歇一会儿再判断，避免连续搜索却不说话。"""
        moment = time.monotonic() if now is None else now
        with self._lock:
            state = self._state(room)
            state["retry_after"] = moment + RETRY_SECONDS
            state["due"] = None

    def heard(self, room, now=None):
        """群里有人说了一句。等安静几秒再判断要不要接，不限机器人是不是刚说过话。"""
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._state(room)["due"] = moment + QUIET_SECONDS

    def take(self, room, *, minutes, limit, last_id, quiet, now=None):
        """该判断时返回 "hot" 或 "beat"，否则返回 None。

        last_id 是最近一条人类发言的编号，quiet 表示它已经过了防抖时间。"""
        moment = time.monotonic() if now is None else now
        with self._lock:
            state = self._state(room)
            state["beats"] = [item for item in state["beats"] if moment - item < HOUR]
            if state["retry_after"] is not None and moment < state["retry_after"]:
                return None
            if state["due"] is not None:
                if moment < state["due"]:
                    return None
                state["due"] = None
                state["seen"] = last_id
                # 安静后再接。这一小时主动说过的话已经到上限时，这次也不接。
                if len(state["beats"]) >= limit:
                    return None
                return "hot"
            if minutes <= 0 or len(state["beats"]) >= limit:
                return None
            if state["glanced"] is None:
                # 刚启动时先记下时间，满一个间隔再看，避免把重启前的聊天当成新发言。
                state["glanced"], state["seen"] = moment, last_id
                return None
            if moment - state["glanced"] < minutes * 60 or last_id is None or last_id == state["seen"] or not quiet:
                return None
            state["glanced"], state["seen"] = moment, last_id
            return "beat"
