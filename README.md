# astrbot_plugin_voice_hearing

把 QQ 语音条转成**多模态模型能吃的音频输入**。

## 问题

QQ 语音条的格式（通常是 SILK 或 AMR）大多数模型接口都不认。用户在群里发一段语音，机器人只能看到 `[语音]` 三个字。

## 做法

```
收到语音消息
        ↓
下载原始音频
        ↓
ffmpeg 转码：SILK/AMR → wav/mp3（单声道 16k）
        ↓
给模型三选一：
  ├─ 接口支持音频输入 → 直接传（OpenAI 兼容层的 input_audio）
  ├─ 支持文件上传     → 走 files 接口
  └─ 都不支持         → 降级：本地 whisper 转文字
```

## 为什么要有降级链

「模型支持音频」这件事要实测。很多中转站声称支持，实际请求会返回「没有收到音频」或者直接忽略这个字段。

所以插件不赌，按能力逐级降级，并在日志里写清楚走了哪条路。

## 配置

```json
{
  "mode": "auto",
  "whisper_model": "base",
  "max_seconds": 120
}
```

- `mode`：`auto`（自动探测）/ `api`（直接传音频）/ `text`（本地转文字）
- `max_seconds`：超过这个时长不处理

## 依赖

- `ffmpeg`
- `faster-whisper` 或 `openai-whisper`（降级路径需要）

## 许可

MIT