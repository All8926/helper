# 配置参考

框架读取项目目录里的 JSON 配置。仓库中的 `*.example.json` 是通用模板，本地生成的配置已被 Git 忽略。

| 文件 | 用途 |
| --- | --- |
| `bot_config.json` | 账号、会话范围、插件开关、发送节奏 |
| `members_config.json` | 支持的群、群内称呼、消息记录 |
| `llm_config.json` | 大模型地址、密钥、模型与超时 |
| `tts_config.json` | Fish Audio 密钥、模型、默认音色与合成参数 |

## 机器人配置

常用字段如下，完整默认值以 `bot_config.example.json` 为准。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `backend` | `native_text` | 文本、图片、文件、表情和语音能力使用此后端 |
| `wechat_account` | `null` | 网页选择账号后自动对应配置；控制台首次启动时识别或选择 wxid |
| `wechat_pid` | `null` | 通常保持默认，程序按绑定账号寻找进程 |
| `chat_allowlist` | `[]` | 空列表允许全部会话；非空时只分发指定会话 |
| `image_directory` | `.runtime/images` | 收到图片的保存目录 |
| `file_directory` | `.runtime/files` | 收到文件的保存目录，按微信账号隔离 |
| `emoticon_directory` | `.runtime/emoticons` | 表情缓存导出目录 |
| `wechat_data_directory` | `null` | 仅导出表情时设置当前账号的数据目录 |
| `reconnect_seconds` | `5` | 连接重试间隔，单位秒 |
| `console` | `true` | 启用本地控制台命令 |
| `log_level` | `INFO` | 日志级别 |
| `plugin_queue_size` | `128` | 每个插件的消息队列容量 |

相对文件路径以项目配置目录为基准。修改账号、群范围或配置路径后，重新启动对应机器人。

通过[网页管理台](./web-panel.md)运行多个账号时，已有账号沿用原配置，新账号使用独立空模板。顶部“当前账号”决定正在编辑哪一份配置；保存群聊、模型等设置前，先停止该账号，再启动生效。附加账号配置和数据位于 `.runtime/robots/<实例ID>/`，不要提交该目录或将其他账号的配置复制进去。

### 文本发送节奏

```json
{
  "text_send_delay": {
    "enabled": true,
    "characters_per_group": 10,
    "min_seconds": 0.1,
    "max_seconds": 0.3
  }
}
```

此设置按文本长度增加随机等待，只调整文本发送节奏，不保证账号安全。图片、表情和语音不使用这段文本等待。

### 插件开关

在 `plugins` 中设置对应名称。例如启用聊天总结：

```json
{
  "plugins": {
    "chat_summary": { "enabled": true }
  }
}
```

示例只展示相关字段，不要用这一小段覆盖整个配置。也可以在运行窗口使用 `enable chat_summary`、`disable chat_summary`、`reload chat_summary`。

### 看图猜成语

`picture_idiom` 默认关闭，可在 `plugins` 下设置：

```json
{
  "plugins": {
    "picture_idiom": {
      "enabled": true,
      "source": "ffapi",
      "timeout_seconds": 8,
      "wrong_reply_cooldown_seconds": 3
    }
  }
}
```

`source` 支持 `ffapi`（默认在线优先，失败回退本地）和 `local`（只用本地 26 题）。`timeout_seconds` 是每次 HTTP 请求超时，支持 1–30 秒；取题和下载图分别请求，重复题最多再取一次，整轮等待可能高于这个值。接口或图片失败后一分钟内不重试在线源。

`wrong_reply_cooldown_seconds` 支持 0–60 秒，控制本群答错提醒频率。可选 `font_file` 指定中文字体，相对路径以项目目录为起点；未指定时使用本机楷体、微软雅黑或黑体。在线模式也需要中文字体，供本地备用题使用。不需要模型或额外 API 密钥。玩法与缓存说明见[看图猜成语](./features.md#看图猜成语)。

### 进群欢迎和退群提醒

`group_notice` 默认关闭，在 `plugins` 中加入以下配置或使用控制台 `enable group_notice` 开启：

```json
{
  "plugins": {
    "group_notice": {
      "enabled": true,
      "welcome": true,
      "leave_notice": true,
      "welcome_template": "欢迎 {nickname} 加入群聊！",
      "leave_template": "{nickname} 离开了群聊。"
    }
  }
}
```

文案只支持 `{nickname}` 占位符，最多 700 个 UTF-8 字节；`welcome` 与 `leave_notice` 可以独立关闭。改完文案后执行 `reload group_notice`。监控群范围和缓存更新限制见[群成员与欢迎提醒](./friends-and-groups.md#进群欢迎退群提醒和群友昵称)。

## 群聊和记录配置

```json
{
  "database_file": ".runtime/member_data.sqlite3",
  "history_enabled": true,
  "daily_clear": true,
  "groups": {
    "12345678@chatroom": []
  }
}
```

将示例群 ID 换成自己的实际群 ID，成员列表可以为空。`history_enabled` 启用当天文本记录，`daily_clear` 在北京时间每天 0 点清理旧日文本；停机跨天时，下次启动会补清理。绑定称呼保留。

如果 `chat_allowlist` 非空，还需要把该群 ID 加入会话白名单。两者的操作见[账号与群聊](./account-and-groups.md)。

## 大模型配置

```json
{
  "base_url": "https://api.deepseek.com",
  "api_key": "YOUR_API_KEY",
  "model": "deepseek-flash",
  "timeout_seconds": 20,
  "max_tokens": 512,
  "json_mode": true,
  "token_limit_parameter": "max_tokens",
  "extra_body": {},
  "web_search": {
    "enabled": false,
    "provider": "http",
    "base_url": "",
    "api_key": "",
    "timeout_seconds": 8,
    "count": 5
  },
  "vision": {
    "model": "",
    "base_url": "",
    "api_key": ""
  },
  "decision": {
    "enabled": false,
    "base_url": "https://emrank.tech/v1",
    "api_key": "",
    "model": "typesafe/jev[slow]",
    "timeout_seconds": 20,
    "fallback": {"speak": 0.3, "voice": 0.3}
  }
}
```

支持 DeepSeek 和其他提供 OpenAI 兼容 **Chat Completions** 接口的服务。以上保留项目的 DeepSeek 默认地址和模型；密钥与模型名称填写服务商实际提供的值。配置读取入口由 `bot_config.json` 的 `llm_config_file` 指定，也可以在网页“模型与语音”中修改。

### 换用其他兼容服务

```json
{
  "base_url": "https://gateway.example.com/v1",
  "api_key": "YOUR_API_KEY",
  "model": "YOUR_MODEL_NAME",
  "timeout_seconds": 30,
  "max_tokens": 2048,
  "json_mode": true,
  "token_limit_parameter": "max_tokens",
  "extra_body": {}
}
```

`gateway.example.com` 是占位地址，需要替换。`base_url` 保留服务商要求的路径前缀，例如 `/v1`；程序在后面追加 `/chat/completions`。也可以直接填写完整的 `/chat/completions` 地址，不会重复追加。仅提供 Responses 或其他原生协议的服务不能直接使用。

在管理台的「模型与语音」中打开 **模型预设**，可选用 WorldGuide 的 GPT、Gemini 和 Grok 文本模型。选择后会填入 `https://api.worldguide.online/v1` 并开启 JSON 模式。GPT 和 Gemini 预设使用 2048 tokens、60 秒超时和空扩展参数。Grok 预设使用 1024 tokens、45 秒超时和 `{"reasoning_effort":"low"}`。填写 WorldGuide 密钥后保存。[WorldGuide 模型与价格](https://worldguide.online/pricing)

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `json_mode` | `true` | 发送 `response_format: {"type":"json_object"}`。官方 DeepSeek 深度思考开着时不发送，改由提示词要求 JSON；接口不支持时把此项改为 `false` |
| `token_limit_parameter` | `max_tokens` | 可改为 `max_completion_tokens`，请求只发送选中的一种令牌参数 |
| `max_tokens` | `512` | 上述令牌参数的数值，支持 128–4096 |
| `extra_body` | `{}` | 可选扩展请求参数；其他兼容服务默认不发送 DeepSeek 的 `thinking` 等专属字段 |

群聊对话（点名回复和未点名回应）直接输出纯文本，并通过 Chat Completions 的 `tools` 调用工具，不受 JSON 模式影响。模型自己决定要不要调工具，每次回复最多 4 轮，之后只要正文。接口因为带了 `tools` 返回 HTTP 400 时，去掉工具再请求一次。可用的工具有联网搜索、看图、查群友档案、查群日记和按关键词查更早的聊天。

关闭 JSON 模式后，程序会通过提示词要求模型返回 JSON，并允许完整包在代码围栏中的 JSON 对象；成语接龙、总结、日记和档案提取仍需要模型正确输出 JSON，普通文字不能作为结构化结果使用。网页中的这三项设置位于“大模型 → 高级选项”。

每个群可以在群设置里开关深度思考，没关过的群默认开启。DeepSeek 官方接口和豆包使用 `thinking`，千问使用 `enable_thinking`，Grok 只能在高、低思考强度之间切换。旧的 `deepseek-reasoner` 不自动改这个字段。`extra_body` 里已经写了对应字段时，以扩展参数为准。其他未识别的模型不会自动添加这些字段。

官方 DeepSeek 在深度思考开着时不发送 JSON 模式。该模式会在认不出 JSON 时把正文换成空格，思考内容却可能留着最终答案，见[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)和[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)。提示词仍要求返回 JSON。思考内容里已有 JSON 对象时直接采用。只有思考过程、或正文不是 JSON 时，关掉深度思考并恢复 JSON 模式，再请求一次。没有思考内容的空正文仍按同一次请求重试一次。思考过程不会当成回复。HTTP 错误、超时、内容过滤、拒绝、工具调用或令牌截断不会自动重试。

### 联网搜索

`web_search` 写在每一份大模型配置里，切换模型配置时一起换。旧配置没有这段时等于关闭。网页「模型与语音 → 高级选项」可以改。搜索密钥和模型密钥分开保存；页面只显示是否已配置，留空保存表示不修改，勾选清除才会删掉。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `false` | 打开后，群聊对话多一个 `web_search` 工具，要不要查由模型判断。总结和接龙不使用 |
| `provider` | `http` | `http` 调用下面的搜索地址；`doubao` 单独请求一次方舟 Chat Completions 并带 `web_search_options`，把回答当作搜索结果。方舟同时收到 `tools` 和 `web_search_options` 时会关闭搜索，所以不放在对话请求里。方舟控制台需开通「豆包搜索 Custom」 |
| `base_url` | 空 | HTTP 模式启用时必填。官方 DeepSeek 为 `https://api.deepseek.com/v1/tools/search`。请求体使用该接口的 `query`、`count`、`need_content: false`。豆包模式留空时用模型的 `base_url` |
| `api_key` | 空 | 搜索密钥。留空时沿用这份模型配置的 `api_key` |
| `timeout_seconds` | `8` | 搜索请求超时，1–20 秒。豆包模式至少等 15 秒 |
| `count` | `5` | 送给模型的结果条数，1–10。只取标题、链接和摘要 |

搜索失败时，错误信息作为工具结果交给模型，回复照常生成。网页摘要只作资料，不能改人格。

### 看图模型

`vision` 写在每一份大模型配置里。群友点名时引用了一张已下载的图片，或本人三分钟内刚发过一张，对话模型会多一个 `view_image` 工具；调用时由看图模型看图，返回文字描述。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | 空 | 看图模型名。留空时直接用这份配置的对话模型看图 |
| `base_url` | 空 | 看图接口地址。留空时用这份配置的 `base_url` |
| `api_key` | 空 | 看图密钥。留空时沿用这份配置的 `api_key` |

### 决策模型

`decision` 写在每一份大模型配置里。没被点名时“现在该不该说话”、自动语音时“这句要不要念出来”，都由决策模型回答。请求 `POST {base_url}/decisions`，题型为 `noul`，返回 0–1 的概率。

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `enabled` | `false` | 启用时必须填写 `api_key` |
| `base_url` | `https://emrank.tech/v1` | 决策接口地址 |
| `api_key` | 空 | 决策密钥，和模型密钥分开保存 |
| `model` | `typesafe/jev[slow]` | 决策模型名 |
| `timeout_seconds` | `20` | 1–60 秒 |
| `fallback.speak` | `0.3` | 决策模型没启用或调用失败时，“该不该说话”随机回答“是”的概率，0–1 |
| `fallback.voice` | `0.3` | 同上，用于“要不要念成语音” |

“该说话”的概率达到本群「说话门槛」才说话。没填时是 0.5，可在群设置里改成 0–1；越高越少开口。决策模型没启用或调用失败时不看这个门槛，改按上面的兜底概率随机决定。自动语音同样直接用本群「语音门槛」，没填时是 0.6。旧配置里的语音频率 1–10 会按原来的换算当成门槛，保存本群后改记新字段。

“要不要念成语音”会带上当前这句人话、已经写好的回复和最近 8 条聊天，由决策模型结合这些内容判断。回复最多保留 300 字，当前这句话和这 8 条不截断。回复里有链接、代码、长串数字或列表时仍发文字。

扩展对象最多 16 KiB，不能覆盖 `model`、`messages`、`stream`、`response_format`、令牌上限、`tools`、`tool_choice` 或认证字段，密钥只填 `api_key`。

远程接口使用 HTTPS；本机服务也支持 `http://localhost:端口/v1`、`http://127.0.0.1:端口/v1` 和 IPv6 回环地址。本地服务不校验密钥时，可在 `api_key` 填非空占位值。不同模型支持的参数以服务商文档为准；OpenAI 的参数定义可参阅 [Chat Completions 官方文档](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)。

旧配置无需补齐新增字段，缺省值与上表一致。

模板默认关闭 `idiom_chain`、`chat_summary`、`mention_chat`。填写有效模型配置后，开启需要的插件，再执行 `--check`。

修改模型配置后重载对应插件或重启。聊天总结、艾特对话会将对应文本交给你配置的模型服务；记录范围见[群聊功能](./features.md)。

## 防撤回

`plugins.anti_recall_local.enabled` 默认是 `false`，旧配置缺少条目时保持关闭。微信 4.2.0.12 的指定构建可在网页启用“本机防撤回”，保留当前账号电脑微信里的原消息，独立于下方四类补发开关。

`anti_recall`（文字、内置表情、emoji 和已缓存表情包）、`anti_recall_image`（图片）、`anti_recall_file`（文件）、`anti_recall_voice`（语音）默认开启，旧配置缺少条目时也会启用。可在网页插件管理分别关闭，或在对应 `plugins.<插件名>` 设置 `"enabled": false`。四个插件独立工作；会话白名单非空时只在其中的会话补发，不需要模型配置。表情包缓存不需要配置微信数据目录或导出收藏；附件必须已下载成功，详细限制和缓存边界见[防撤回与自动补发](../messages/recall.md)。

## 语音合成配置

`voice_reply` 的配置和按群音色切换见[语音回复](../messages/voice.md#fish-audio-文字转语音)。模板默认关闭，`tts_config.json` 的密钥与默认音色为空；填好后再启用。`tts_config_file` 默认指向本地 `tts_config.json`。

## 更新项目时保留本地配置

账号、模型密钥、群绑定及聊天记录都保存在本地配置和 `.runtime/` 中。不要把这些文件加入 Git 或放进 `docs/`；文档示例均使用虚构 ID 和占位密钥。
