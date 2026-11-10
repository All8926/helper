"""只处理框架确认的成员变化，不响应群里的同名文字。"""
import contextvars
from string import Formatter

# 当前这条进退群事件要使用的文案。没有时 notice_text 用账号模板。
_notice_templates = contextvars.ContextVar("group_notice_templates", default=None)


def _template(value, field):
    if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > 700 or "\0" in value:
        raise ValueError(f"{field} 应为非空文本，最多 700 个 UTF-8 字节")
    try:
        for _, name, spec, conversion in Formatter().parse(value):
            if name is not None and (name != "nickname" or spec or conversion):
                raise ValueError("只支持 {nickname} 占位符")
    except ValueError as error:
        raise ValueError(f"{field} 模板格式错误：{error}") from None
    return value


class Plugin:
    def on_load(self, ctx):
        self.welcome = ctx.config.get("welcome", True)
        self.leave_notice = ctx.config.get("leave_notice", True)
        if type(self.welcome) is not bool or type(self.leave_notice) is not bool:
            raise ValueError("welcome / leave_notice 必须是布尔值")
        self.welcome_template = _template(ctx.config.get("welcome_template", "欢迎 {nickname} 加入群聊！"),
                                         "welcome_template")
        self.leave_template = _template(ctx.config.get("leave_template", "{nickname} 离开了群聊。"),
                                       "leave_template")

    def on_message(self, message, ctx):
        event = message.data.get("group_event", {})
        if message.type != 10002 or not message.is_group or event.get("room_id") != message.chat_id:
            return
        kind = event.get("kind")
        if not event.get("member_id") or kind not in ("join", "leave"):
            return
        welcome, leave_notice, templates = self._for_room(ctx, message)
        if (kind == "join" and not welcome) or (kind == "leave" and not leave_notice):
            return
        nickname = event.get("nickname") or "群友"
        if getattr(ctx, "_members", None) is not None:
            # /绑定 只在本群有效；退出时仍能使用保存过的群昵称。
            nickname = ctx.members.display_name(event["member_id"], message.chat_id) or nickname
        token = _notice_templates.set(templates)
        try:
            ctx.reply(message, self.notice_text(nickname, kind))
        finally:
            _notice_templates.reset(token)

    def _for_room(self, ctx, message):
        """本群没单独写的开关和文案，继续用插件管理里的账号配置。"""
        welcome, leave_notice = self.welcome, self.leave_notice
        templates = (self.welcome_template, self.leave_template)
        if not hasattr(ctx, "room_config"):
            return welcome, leave_notice, templates
        options = ctx.room_config(message)
        if type(options.get("welcome", welcome)) is bool:
            welcome = options.get("welcome", welcome)
        if type(options.get("leave_notice", leave_notice)) is bool:
            leave_notice = options.get("leave_notice", leave_notice)
        try:
            templates = (_template(options.get("welcome_template", self.welcome_template), "welcome_template"),
                         _template(options.get("leave_template", self.leave_template), "leave_template"))
        except ValueError:
            ctx.logger.warning("本群进退群文案无效，已改用账号模板")
        return welcome, leave_notice, templates

    def notice_text(self, nickname, kind):
        """kind 为 join（进群）或 leave（退群），返回通知文字。"""
        override = _notice_templates.get()
        if override is None:
            template = self.welcome_template if kind == "join" else self.leave_template
        else:
            template = override[0] if kind == "join" else override[1]
        return template.format(nickname=nickname[:64])
