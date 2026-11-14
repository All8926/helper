"""把本机已下载的一张图片收成模型能看的 data URL。不访问网络。"""
from __future__ import annotations

import base64
from io import BytesIO
import secrets
import threading
import time

from PIL import Image, UnidentifiedImageError

from .images import validate_image_path

# 边长和体积先压小，避免一次看图把请求撑过超时。
MAX_EDGE = 1280
MAX_INLINE_BYTES = 1536 * 1024
_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _mime(raw):
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _url(mime, raw):
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode("ascii")


def _jpeg_bytes(image):
    """过大或不受支持的图压成一张 JPEG。失败返回空字节。"""
    image.seek(0)
    frame = image.convert("RGB")
    edge, quality = MAX_EDGE, 80
    for _ in range(3):
        copy = frame.copy()
        copy.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        output = BytesIO()
        copy.save(output, format="JPEG", quality=quality)
        data = output.getvalue()
        if 0 < len(data) <= MAX_INLINE_BYTES:
            return data
        edge = max(640, edge // 2)
        quality = 55
    return b""


def vision_data_url(path):
    """返回 data URL。文件无效、过大或无法解码时返回空字符串。"""
    try:
        raw = validate_image_path(path).read_bytes()
        mime = _mime(raw)
        with Image.open(BytesIO(raw)) as image:
            image.load()
            frames = getattr(image, "n_frames", 1) or 1
            edge = max(image.size)
            if mime in _MIMES and edge <= MAX_EDGE and len(raw) <= MAX_INLINE_BYTES and (
                    frames == 1 or mime in ("image/gif", "image/webp")):
                return _url(mime, raw)
            jpeg = _jpeg_bytes(image)
        return _url("image/jpeg", jpeg) if jpeg else ""
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return ""


def message_id(value):
    """微信消息编号。0、布尔值和超范围整数都不当作编号。"""
    if isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 20:
        try:
            value = int(value)
        except ValueError:
            return ""
    if type(value) is int and 0 < value < 2**64:
        return str(value)
    return ""


class SeenImages:
    """记住本进程里刚下载好的群图片，供点名时对上那一张。"""

    def __init__(self, *, window=180, limit=200):
        if type(window) is not int or window < 0 or type(limit) is not int or limit < 1:
            raise ValueError("图片记忆的时间和条数无效")
        self.window = window
        self.limit = limit
        self._lock = threading.Lock()
        self._by_id = {}
        self._by_token = {}
        self._order = []

    def remember(self, room, sender, identifier, path):
        """记下已下载的图，返回给模型的编号。无效输入返回空字符串。"""
        identifier = message_id(identifier)
        if not identifier or not all(isinstance(value, str) and value and "\0" not in value
                                      for value in (room, sender, path)):
            return ""
        with self._lock:
            current = self._by_id.get(identifier)
            token = current[4] if current and len(current) > 4 else secrets.token_hex(4)
            self._by_id[identifier] = (room, sender, path, time.monotonic(), token)
            self._by_token[token] = identifier
            if identifier in self._order:
                self._order.remove(identifier)
            self._order.append(identifier)
            while len(self._order) > self.limit:
                dropped = self._order.pop(0)
                old = self._by_id.pop(dropped, None)
                if old and len(old) > 4:
                    self._by_token.pop(old[4], None)
        return token

    def by_id(self, identifier):
        identifier = message_id(identifier)
        with self._lock:
            row = self._by_id.get(identifier)
        return row[2] if row else ""

    def token_for(self, identifier):
        identifier = message_id(identifier)
        with self._lock:
            row = self._by_id.get(identifier)
        return row[4] if row and len(row) > 4 else ""

    def path(self, token):
        """只接受程序分配的编号。路径和认不出的编号都打不开。"""
        if not isinstance(token, str) or len(token) != 8 or any(ch not in "0123456789abcdef" for ch in token):
            return ""
        with self._lock:
            identifier = self._by_token.get(token)
            row = self._by_id.get(identifier) if identifier else None
        return row[2] if row else ""

    def wait_ready(self, identifier, timeout=5):
        """文件一到就返回。没在下载就立刻返回。到上限还没有也返回。"""
        from .image_receiver import image_download_pending
        deadline = time.monotonic() + timeout
        while True:
            if self.by_id(identifier):
                return True
            if not image_download_pending(identifier) or time.monotonic() >= deadline:
                return bool(self.by_id(identifier))
            time.sleep(0.05)

    def clear(self):
        with self._lock:
            self._by_id.clear()
            self._by_token.clear()
            self._order.clear()

    def latest(self, room, sender, now=None):
        moment = time.monotonic() if now is None else now
        with self._lock:
            rows = [row for row in self._by_id.values()
                    if row[0] == room and row[1] == sender and moment - row[3] <= self.window]
        if not rows:
            return ""
        return max(rows, key=lambda row: row[3])[2]


# 本进程已下载图片的编号和路径。提示词里只出现编号。
catalog = SeenImages(window=180, limit=200)
