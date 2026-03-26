import base64
import json
import mimetypes
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError


@dataclass
class ScriptAssets:
    numbered: Dict[str, Path]


def _parse_numbered_assets(md: str) -> ScriptAssets:
    numbered: Dict[str, Path] = {}
    for line in md.splitlines():
        m = re.match(r"^-\s*\[图(\d+)\].*：\s*(/\S+)$", line.strip())
        if not m:
            continue
        idx = m.group(1)
        numbered[idx] = Path(m.group(2))
    return ScriptAssets(numbered=numbered)


def _extract_section(md: str, header: str) -> str:
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i + 1
            break
    if start is None:
        return ""
    out: List[str] = []
    for line in lines[start:]:
        if re.match(r"^##\s+", line.strip()):
            break
        out.append(line)
    return "\n".join(out).strip()


def _extract_timeblock(md: str, header: str) -> str:
    lines = md.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i + 1
            break
    if start is None:
        return ""
    out: List[str] = []
    for line in lines[start:]:
        if re.match(r"^###\s+", line.strip()):
            break
        if re.match(r"^##\s+", line.strip()):
            break
        out.append(line)
    return "\n".join(out).strip()


def _extract_quotes(block: str) -> List[str]:
    quotes = re.findall(r"“([^”]+)”", block)
    return [q.strip() for q in quotes if q.strip()]


def _to_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0]
    if not mime:
        ext = path.suffix.lower()
        if ext == ".png":
            mime = "image/png"
        elif ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            mime = "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class ArkVerboseClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    ):
        if not api_key:
            raise ValueError("ARK_API_KEY is required")
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def request_json(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = self.base_url + path
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = request.Request(url, method=method, headers=self._headers, data=data)
        try:
            with request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as e:
            raise RuntimeError(e.read().decode("utf-8", errors="replace"))


def _log_body_for_humans(body: dict) -> dict:
    def scrub(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "url" and isinstance(v, str) and v.startswith("data:"):
                    out[k] = "<data_uri omitted>"
                else:
                    out[k] = scrub(v)
            return out
        if isinstance(obj, list):
            return [scrub(x) for x in obj]
        return obj

    return scrub(body)


def _wait_task(client: ArkVerboseClient, task_id: str, timeout_s: int = 3600) -> dict:
    start = time.time()
    while True:
        resp = client.request_json("GET", f"/contents/generations/tasks/{task_id}")
        status = resp.get("status")
        print(f"[poll] task_id={task_id} status={status}")
        if status == "succeeded":
            return resp
        if status in {"failed", "expired"}:
            raise RuntimeError(f"task ended: {resp.get('error')}")
        if time.time() - start > timeout_s:
            raise RuntimeError("timeout waiting for task")
        time.sleep(10)


def _concat(ffmpeg: str, seg_paths: List[Path], out_path: Path) -> None:
    list_path = out_path.with_suffix(".concat.txt")
    list_path.write_text(
        "\n".join([f"file '{p.name}'" for p in seg_paths]) + "\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                str(out_path),
            ],
            check=True,
        )
    list_path.unlink(missing_ok=True)


def _load_from_llm_env(
    md_path: str = "/Users/bytedance/Desktop/Workspace/code/LLM_env.md",
) -> Tuple[Optional[str], Optional[str]]:
    path = Path(md_path)
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    api_key = None
    model = None
    m = re.search(r"API Key:\s*([^\s]+)", text)
    if m:
        api_key = m.group(1)
    m2 = re.search(r"doubao-seedance-2-0-260128\)\s*\nEndpoint:\s*(ep-[\w-]+)", text)
    if m2:
        model = m2.group(1)
    return api_key, model


def generate_from_script(
    script_path: str,
    output_path: str,
    model: str,
    api_key: str,
    seed: int = 2026,
    resolution: str = "720p",
    ratio: str = "16:9",
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
) -> str:
    script_file = Path(script_path)
    md = script_file.read_text(encoding="utf-8")
    assets = _parse_numbered_assets(md)

    need = ["1", "2", "3", "4", "6", "7", "8"]
    missing = [k for k in need if k not in assets.numbered]
    if missing:
        raise RuntimeError(f"missing numbered assets in script: {missing}")

    for k, p in assets.numbered.items():
        if not p.exists():
            raise RuntimeError(f"asset missing: [图{k}] {p}")

    block_constraints = _extract_section(md, "## 不可变约束（强一致性）")
    block_0_15 = _extract_timeblock(md, "### 0–15s（外观 + 贴墙 + 中开门）")
    block_15_30 = _extract_timeblock(md, "### 15–30s（人物亲自开门 + 内部空间 + 藏肉）")

    constraints_bullets: List[str] = []
    for line in block_constraints.splitlines():
        if line.strip().startswith("- "):
            constraints_bullets.append(line.strip()[2:].strip())
    constraints = "；".join([x for x in constraints_bullets if x])

    q1 = " ".join(_extract_quotes(block_0_15)) or ""
    q2 = " ".join(_extract_quotes(block_15_30)) or ""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seg1 = out.with_name(out.stem + "_seg1.mp4")
    seg2 = out.with_name(out.stem + "_seg2.mp4")

    client = ArkVerboseClient(api_key=api_key, base_url=base_url)

    print("[agent] step=load_script path=", str(script_file))
    print("[agent] step=resolve_assets count=", len(assets.numbered))
    print("[agent] step=build_prompts seed=", seed)

    prompt1 = (
        "你是商业广告导演。请生成写实电影风格的家电广告。\n"
        f"强一致性约束：{constraints}\n"
        "生成要求：30秒一镜到底，但本次为第1/2段（15秒），不允许跳切/换景/换机位/换焦段/换曝光/换色温。\n"
        "拼接要求：最后2秒镜头稳住，构图固定，人物右手握住中间把手停1秒不动，作为下一段的衔接点。\n"
        f"台词（中文口语，连贯）：{q1}\n"
    )

    prompt2 = (
        "你是商业广告导演。请生成写实电影风格的家电广告。\n"
        f"强一致性约束：{constraints}\n"
        "生成要求：30秒一镜到底，但本次为第2/2段（15秒），不允许跳切/换景/换机位/换焦段/换曝光/换色温。\n"
        "衔接要求：开头必须复刻上一段结尾同一构图与姿势，先停0.5-1秒再继续开门动作。\n"
        "物理一致（放肉修复）：只做一次动作链“拿起肉→放入抽屉中部→停1秒→推回抽屉→关门结束”，肉的形态与位置全程保持一致，不穿模、不瞬移、不变形。\n"
        f"台词（中文口语，连贯）：{q2}\n"
    )

    refs1 = [assets.numbered["1"], assets.numbered["7"], assets.numbered["8"], assets.numbered["2"]]
    refs2 = [assets.numbered["1"], assets.numbered["7"], assets.numbered["4"], assets.numbered["6"]]

    def create_task(prompt: str, refs: List[Path], seg_seed: int) -> Tuple[str, dict]:
        content: List[dict] = [{"type": "text", "text": prompt}]
        for p in refs:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _to_data_uri(p)},
                    "role": "reference_image",
                }
            )
        body = {
            "model": model,
            "content": content,
            "duration": 15,
            "ratio": ratio,
            "resolution": resolution,
            "watermark": False,
            "seed": seg_seed,
            "generate_audio": True,
        }
        print("[agent] step=seedance_create method=POST path=/contents/generations/tasks")
        print("[agent] request_body=", json.dumps(_log_body_for_humans(body), ensure_ascii=False))
        create = client.request_json("POST", "/contents/generations/tasks", body=body)
        task_id = create.get("id")
        if not task_id:
            raise RuntimeError(f"no task id: {create}")
        return task_id, body

    print("[agent] step=segment1 duration=15s")
    task1, body1 = create_task(prompt1, refs1, seed)
    final1 = _wait_task(client, task1)
    url1 = (final1.get("content") or {}).get("video_url")
    if not url1:
        raise RuntimeError("segment1 missing video_url")
    print("[agent] step=download segment1 url=", url1)
    with request.urlopen(url1, timeout=60) as resp:
        seg1.write_bytes(resp.read())
    print("[agent] step=segment1_saved path=", str(seg1), "bytes=", seg1.stat().st_size)

    print("[agent] step=segment2 duration=15s")
    task2, body2 = create_task(prompt2, refs2, seed)
    final2 = _wait_task(client, task2)
    url2 = (final2.get("content") or {}).get("video_url")
    if not url2:
        raise RuntimeError("segment2 missing video_url")
    print("[agent] step=download segment2 url=", url2)
    with request.urlopen(url2, timeout=60) as resp:
        seg2.write_bytes(resp.read())
    print("[agent] step=segment2_saved path=", str(seg2), "bytes=", seg2.stat().st_size)

    print("[agent] step=concat segments=2 duration=30s")
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_bin = get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = os.environ.get("FFMPEG", "ffmpeg")
    print("[agent] ffmpeg=", ffmpeg_bin)
    _concat(ffmpeg_bin, [seg1, seg2], out)

    trace_path = out.with_suffix(".trace.json")
    trace_path.write_text(
        json.dumps(
            {
                "script": str(script_file),
                "output": str(out),
                "model": model,
                "seed": seed,
                "segments": [
                    {"task_id": task1, "request": _log_body_for_humans(body1)},
                    {"task_id": task2, "request": _log_body_for_humans(body2)},
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[agent] trace=", str(trace_path))

    seg1.unlink(missing_ok=True)
    seg2.unlink(missing_ok=True)

    print("[agent] step=done output=", str(out))
    return str(out)


if __name__ == "__main__":
    api_key = os.getenv("ARK_API_KEY")
    model = os.getenv("ARK_VIDEO_MODEL")
    if not api_key or not model:
        fallback_key, fallback_model = _load_from_llm_env()
        api_key = api_key or fallback_key
        model = model or fallback_model

    if not api_key:
        raise SystemExit("ARK_API_KEY env is required (or provide it in LLM_env.md)")
    if not model:
        raise SystemExit("ARK_VIDEO_MODEL env is required (or provide Seedance endpoint in LLM_env.md)")

    script = os.getenv(
        "SCRIPT_PATH",
        "/Users/bytedance/Desktop/Workspace/code/Director/cases/test3/剧本.md",
    )
    out = os.getenv(
        "OUTPUT_PATH",
        "/Users/bytedance/Desktop/Workspace/code/Director/cases/test3/haier_onetake_30s_agent_trace.mp4",
    )
    seed = int(os.getenv("SEED", "2026"))
    generate_from_script(
        script_path=script,
        output_path=out,
        model=model,
        api_key=api_key,
        seed=seed,
        resolution=os.getenv("RESOLUTION", "720p"),
        ratio=os.getenv("RATIO", "16:9"),
    )
