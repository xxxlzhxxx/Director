import json
import time
from typing import Any, Dict, Optional
from urllib import request
from urllib.error import HTTPError, URLError


class ArkContentGenerationClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        timeout_s: int = 60,
    ):
        if not api_key:
            raise ValueError("ARK_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        req = request.Request(url=url, method=method, headers=headers, data=data)
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ark API error {e.code}: {raw}")
        except URLError as e:
            raise RuntimeError(f"Ark API network error: {e}")

    def create_task(self, model: str, content: list, **params) -> str:
        body = {"model": model, "content": content, **params}
        resp = self._request("POST", "/contents/generations/tasks", body=body)
        task_id = resp.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"Unexpected create task response: {resp}")
        return task_id

    def get_task(self, task_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/contents/generations/tasks/{task_id}")

    def wait_task(
        self,
        task_id: str,
        timeout_s: int = 900,
        poll_interval_s: int = 10,
    ) -> Dict[str, Any]:
        start = time.time()
        while True:
            resp = self.get_task(task_id)
            status = resp.get("status")
            if status == "succeeded":
                return resp
            if status in {"failed", "expired"}:
                raise RuntimeError(f"Task ended with status={status}: {resp.get('error')}")
            if time.time() - start > timeout_s:
                raise RuntimeError("Timeout waiting for Ark video generation task")
            time.sleep(poll_interval_s)

    def download(self, url: str, save_at: str) -> None:
        with request.urlopen(url, timeout=self.timeout_s) as resp, open(save_at, "wb") as f:
            f.write(resp.read())


class ArkVideoGenerationTool:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    ):
        self.client = ArkContentGenerationClient(api_key=api_key, base_url=base_url)

    def text_to_video(
        self,
        prompt: str,
        save_at: str,
        duration: float,
        config: dict,
    ) -> Dict[str, Any]:
        model = config.get("model")
        if not model:
            raise ValueError("Missing Ark model (endpoint id). Set config.model.")

        params: Dict[str, Any] = {
            "duration": int(duration),
            "watermark": bool(config.get("watermark", False)),
        }
        for key in [
            "resolution",
            "ratio",
            "seed",
            "camera_fixed",
            "generate_audio",
            "draft",
            "return_last_frame",
            "service_tier",
            "execution_expires_after",
            "callback_url",
        ]:
            if key in config and config[key] is not None:
                params[key] = config[key]

        content = [{"type": "text", "text": prompt}]
        task_id = self.client.create_task(model=model, content=content, **params)
        final = self.client.wait_task(task_id=task_id)
        video_url = (final.get("content") or {}).get("video_url")
        if not isinstance(video_url, str) or not video_url:
            raise RuntimeError(f"Task succeeded but missing video_url: {final}")
        self.client.download(video_url, save_at=save_at)
        return {"task_id": task_id, "video_url": video_url, "video_path": save_at}

    def image_to_video(
        self,
        image_url: str,
        save_at: str,
        duration: float,
        config: dict,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = config.get("model")
        if not model:
            raise ValueError("Missing Ark model (endpoint id). Set config.model.")

        params: Dict[str, Any] = {
            "duration": int(duration),
            "watermark": bool(config.get("watermark", False)),
        }
        for key in [
            "resolution",
            "ratio",
            "seed",
            "camera_fixed",
            "generate_audio",
            "draft",
            "return_last_frame",
            "service_tier",
            "execution_expires_after",
            "callback_url",
        ]:
            if key in config and config[key] is not None:
                params[key] = config[key]

        content = []
        if isinstance(prompt, str) and prompt.strip():
            content.append({"type": "text", "text": prompt})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
                "role": "first_frame",
            }
        )

        task_id = self.client.create_task(model=model, content=content, **params)
        final = self.client.wait_task(task_id=task_id)
        video_url = (final.get("content") or {}).get("video_url")
        if not isinstance(video_url, str) or not video_url:
            raise RuntimeError(f"Task succeeded but missing video_url: {final}")
        self.client.download(video_url, save_at=save_at)
        return {"task_id": task_id, "video_url": video_url, "video_path": save_at}
