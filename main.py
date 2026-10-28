"""用 Gemini 转述语音条，把文字描述塞给主模型（deepseek）。

长音频按 100 秒切片（保留 1 秒重叠，避免切断句子）逐片转述，再按顺序拼接。
本模块的 transcribe_audio() / VoiceHearingPlugin.transcribe_file() 可被外部直接调用，
/AstrBot/data/scripts/voice_transcribe.py 复用的就是这里的同一套逻辑。
"""

import asyncio
import base64
import json
import os
import uuid
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.message.components import Record

TMP = Path("/tmp/voice")
TMP.mkdir(parents=True, exist_ok=True)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE = "https://api.yoshub.com/v1"
GEMINI_MODEL = "gemini-3.8-flash"

# 切片参数：每片 100 秒，片与片重叠 1 秒（步进 99 秒），防止切断句子
CHUNK_SECONDS = 100
OVERLAP_SECONDS = 1
CHUNK_STEP = CHUNK_SECONDS - OVERLAP_SECONDS

FFMPEG_TIMEOUT = 180          # 单次 ffmpeg 转码/切片超时
GEMINI_TIMEOUT = 100          # 单次 Gemini 调用超时
MAX_TOKENS = 2048             # 100 秒原话照写，500 会截断

PROMPT = (
    "完整描述这段语音。要求："
    "1) 原话照写，一句都不要省略、不要概括；"
    "2) 如果是唱歌：说明是什么歌、谁唱的、歌词逐句写出来；"
    "   并且评价唱功——音准有没有跑调、气息稳不稳、音色什么样、"
    "   有没有破音或刻意炫技，水平大概什么档次；"
    "3) 描述说话人的音色（男/女、年龄感、沙哑还是清亮、口音）、"
    "   语速、音量、情绪（笑/哭/生气/敷衍/兴奋，用括号标注）；"
    "4) 描述背景音：有没有配乐、环境声、其他人的声音；"
    "5) 听不清或拿不准的地方直说听不清，绝对不要编；"
    "6) 评价别客气、别留余地：跑调就说跑调，难听就说难听，"
    "   破音就说破了，五音不全就说五音不全。"
    "   不要用「还算」「不算差」「有待提高」这种软话，"
    "   也不要替他打圆场。好坏直接说，他听得进。"
)


# ---------------------------------------------------------------- 基础工具


async def _run(cmd: list[str], timeout: float) -> tuple[bool, bytes, bytes]:
    try:
        p = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(p.communicate(), timeout=timeout)
        return p.returncode == 0, out, err
    except Exception as exc:
        logger.warning("[voice] 命令异常 " + " ".join(cmd[:6]) + " -> " + repr(exc))
        return False, b"", repr(exc).encode()


async def probe_duration(path: str) -> float:
    """返回音频时长（秒），探测失败返回 0。"""
    ok, out, _ = await _run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ],
        30,
    )
    if not ok:
        return 0.0
    try:
        return max(0.0, float(out.decode("utf-8", "ignore").strip()))
    except Exception:
        return 0.0


def _is_silk(path: Path) -> bool:
    if path.suffix.lower() in (".silk", ".slk"):
        return True
    try:
        with path.open("rb") as f:
            head = f.read(16)
        return b"#!SILK" in head
    except Exception:
        return False


async def _silk_to_wav(src: Path, dst: Path) -> bool:
    """QQ silk -> 24k 单声道 wav（pysilk，ffmpeg 不认 silk）。"""
    try:
        import pysilk
    except Exception as exc:
        logger.warning("[voice] 缺 pysilk，无法解 silk: " + repr(exc))
        return False
    pcm = dst.with_suffix(".pcm")
    try:
        with src.open("rb") as fi, pcm.open("wb") as fo:
            pysilk.decode(fi, fo, 24000)
        ok, _, err = await _run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "s16le", "-ar", "24000", "-ac", "1",
                "-i", str(pcm), "-ac", "1", "-ar", "24000", str(dst),
            ],
            FFMPEG_TIMEOUT,
        )
        if not ok:
            logger.warning("[voice] silk->wav 失败: " + err.decode("utf-8", "ignore")[-150:])
        return ok and dst.exists() and dst.stat().st_size > 0
    except Exception as exc:
        logger.warning("[voice] silk 解码异常: " + repr(exc))
        return False
    finally:
        try:
            pcm.unlink(missing_ok=True)
        except Exception:
            pass


async def to_mp3(src_path: str, dst: Path) -> bool:
    """任意输入（wav/mp3/flac/m4a/amr/silk...）统一转 16k 单声道 mp3。不截断时长。"""
    src = Path(src_path)
    if not src.exists():
        logger.warning("[voice] 文件不存在: " + str(src))
        return False

    real = src
    tmp_wav = None
    if _is_silk(src):
        tmp_wav = dst.with_name(dst.stem + "_src.wav")
        if not await _silk_to_wav(src, tmp_wav):
            return False
        real = tmp_wav

    try:
        ok, _, err = await _run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(real),
                "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dst),
            ],
            FFMPEG_TIMEOUT,
        )
        if not ok:
            logger.warning("[voice] ffmpeg 转码失败: " + err.decode("utf-8", "ignore")[-200:])
            return False
        return dst.exists() and dst.stat().st_size > 0
    finally:
        if tmp_wav is not None:
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass


async def slice_mp3(src: Path, start: float, duration: float, dst: Path) -> bool:
    ok, _, err = await _run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.2f}", "-t", f"{duration:.2f}", "-i", str(src),
            "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dst),
        ],
        FFMPEG_TIMEOUT,
    )
    if not ok:
        logger.warning(
            "[voice] 切片失败 @" + f"{start:.0f}s: " + err.decode("utf-8", "ignore")[-150:]
        )
    return ok and dst.exists() and dst.stat().st_size > 0


async def ask_gemini_file(mp3_path: Path, prompt: str) -> str:
    """单片音频 -> 转述文本；失败返回空串。"""
    try:
        b64 = base64.b64encode(mp3_path.read_bytes()).decode()
    except Exception as exc:
        logger.warning("[voice] 读片失败: " + repr(exc))
        return ""

    payload = json.dumps({
        "model": GEMINI_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": MAX_TOKENS,
    })
    # base64 音频太长，不能作为命令行参数（会 Argument list too long）
    pf = TMP / ("payload_" + uuid.uuid4().hex[:8] + ".json")
    try:
        pf.write_text(payload, encoding="utf-8")
    except Exception as exc:
        logger.warning("[voice] 写 payload 失败: " + repr(exc))
        return ""

    try:
        ok, out, err = await _run(
            [
                "curl", "-s", "-m", "90",
                "-H", "Authorization: Bearer " + GEMINI_KEY,
                "-H", "Content-Type: application/json",
                "-d", "@" + str(pf),
                GEMINI_BASE + "/chat/completions",
            ],
            GEMINI_TIMEOUT,
        )
        text = out.decode("utf-8", "ignore").strip()
        if not ok:
            logger.warning("[voice] curl 失败: " + text[:150] + err.decode("utf-8", "ignore")[:100])
            return ""
        j = json.loads(text)
        if "choices" in j:
            return str(j["choices"][0]["message"].get("content") or "").strip()
        logger.warning("[voice] gemini 返回异常: " + text[:200])
        return ""
    except Exception as exc:
        logger.warning("[voice] gemini exc: " + repr(exc))
        return ""
    finally:
        try:
            pf.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------- 核心入口


async def transcribe_audio(src_path: str, tag: str | None = None) -> str:
    """把任意本地音频文件完整转述成文本（长音频自动切片后按序拼接）。

    Args:
        src_path: 本地音频路径（wav/mp3/flac/m4a/amr/silk 等）。
        tag: 可选，临时文件名前缀，避免并发互踩。

    Returns:
        完整转述文本；整体失败返回空字符串。单片失败用 “[第N段听不清]” 占位。
    """
    src = Path(src_path)
    if not src.exists():
        logger.warning("[voice] 文件不存在: " + str(src))
        return ""

    stamp = (tag or uuid.uuid4().hex[:8]) + "_" + uuid.uuid4().hex[:4]
    full_mp3 = TMP / (stamp + "_full.mp3")
    chunk_files: list[Path] = []

    try:
        if not await to_mp3(str(src), full_mp3):
            return ""

        total = await probe_duration(str(full_mp3))
        if total <= 0:
            total = await probe_duration(str(src))

        starts: list[float] = []
        if total <= CHUNK_SECONDS or total <= 0:
            starts = [0.0]
        else:
            s = 0.0
            while s < total:
                starts.append(s)
                if s + CHUNK_SECONDS >= total:
                    break
                s += CHUNK_STEP

        logger.info(
            "[voice] 开始转述 " + src.name + f"，时长 {total:.1f}s，切 {len(starts)} 片"
        )

        parts: list[str] = []
        for idx, start in enumerate(starts, 1):
            piece = TMP / (stamp + f"_{idx}.mp3")
            chunk_files.append(piece)
            dur = min(CHUNK_SECONDS, max(1.0, total - start)) if total > 0 else CHUNK_SECONDS
            if start <= 0.0 and len(starts) == 1:
                # 单片：直接用整段转码结果，省一次切片
                piece = full_mp3
            elif not await slice_mp3(full_mp3, start, dur, piece):
                parts.append(f"[第{idx}段听不清]")
                continue

            prompt = PROMPT
            if len(starts) > 1:
                prompt = PROMPT + f"\n（这是整段音频的第 {idx}/{len(starts)} 段，接着上下文转述。）"
            text = await ask_gemini_file(piece, prompt)
            logger.info(
                f"[voice] 第 {idx}/{len(starts)} 片 @" f"{start:.0f}s -> {len(text)} 字"
            )
            if text:
                parts.append(text)
            else:
                parts.append(f"[第{idx}段听不清]")

        return "\n".join(p for p in parts if p).strip()
    finally:
        for f in chunk_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            full_mp3.unlink(missing_ok=True)
        except Exception:
            pass


@register(
    "astrbot_plugin_voice_hearing",
    "小肥鱼",
    "用 Gemini 转述 QQ 语音条，把文字描述交给主模型",
    "1.2.0",
)
class VoiceHearingPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.context = context

    async def initialize(self) -> None:
        logger.info("[voice] 已注册：语音条会被 Gemini 转述后交给主模型")

    async def transcribe_file(self, path: str) -> str:
        """给主代理/其他插件直接调用的入口：转述任意本地音频文件，返回完整文本。"""
        return await transcribe_audio(path)

    @filter.on_llm_request()
    async def inject_audio(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        try:
            comps = getattr(getattr(event, "message_obj", None), "message", None) or []
            recs = [c for c in comps if isinstance(c, Record)]
            if not recs:
                return

            umo = str(event.unified_msg_origin or "x")
            safe = "".join(ch if ch.isalnum() else "_" for ch in umo)[-30:]
            descs = []

            for i, rec in enumerate(recs):
                try:
                    src = await rec.convert_to_file_path()
                except Exception as exc:
                    logger.warning("[voice] 取文件失败: " + repr(exc))
                    continue
                if not src or not Path(src).exists():
                    continue

                d = await transcribe_audio(str(src), tag=safe + "_" + str(i))
                logger.info("[voice] 转述结果: " + d[:120])
                if d:
                    descs.append(d)

            if not descs:
                return

            parts = []
            for d in descs:
                parts.append("【语音转述】" + d)
            extra = "\n\n" + "\n\n".join(parts)
            req.prompt = (req.prompt or "") + extra
            logger.info("[voice] 已把 " + str(len(descs)) + " 条转述塞进 prompt")
        except Exception as exc:
            logger.warning("[voice] inject 异常: " + repr(exc))
