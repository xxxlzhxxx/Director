import logging
import os
import json
import uuid

from typing import Optional
from pathlib import Path
import re
import math

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, VideoContent, VideoData, MsgStatus
from director.tools.videodb_tool import VDBVideoGenerationTool, VideoDBTool
from director.tools.ark_video import ArkVideoGenerationTool
from director.tools.stabilityai import (
    StabilityAITool,
    PARAMS_CONFIG as STABILITYAI_PARAMS_CONFIG,
)
from director.tools.fal_video import (
    FalVideoGenerationTool,
    PARAMS_CONFIG as FAL_VIDEO_GEN_PARAMS_CONFIG,
)

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = ["stabilityai", "fal", "videodb", "ark"]

VIDEO_GENERATION_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "Collection ID to store the video",
        },
        "engine": {
            "type": "string",
            "description": "The video generation engine to use. Use Ark by default (Seedance). If the query includes any of the following: 'minimax-video, mochi-v1, hunyuan-video, luma-dream-machine, cogvideox-5b, ltx-video, fast-svd, fast-svd-lcm, t2v-turbo, kling video v 1.0, kling video v1.5 pro, fast-animatediff, fast-animatediff turbo, and animatediff-sparsectrl-lcm'- always use Fal. In case user specifies any other engine, use the supported engines like Stability or Fal.",
            "default": "ark",
            "enum": SUPPORTED_ENGINES,
        },
        "job_type": {
            "type": "string",
            "enum": ["text_to_video", "image_to_video"],
            "description": """
            The type of video generation to perform
            Possible values:
                - text_to_video: generates a video from a text prompt
                - image_to_video: generates a video using an image as a base. The feature is only available in fal engine. Stability AI or VideoDB does not support this feature.
            """,
        },
        "text_to_video": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt to generate the video",
                },
                "name": {
                    "type": "string",
                    "description": "Description of the video generation run in two lines. Keep the engine name and parameter configuration of the engine in separate lines. Keep it short, but show the prompt in full. Here's an example: [Tokyo Sunset - Luma - Prompt: 'An aerial shot of a quiet sunset at Tokyo', Duration: 5s, Luma Dream Machine]",
                },
                "duration": {
                    "type": "number",
                    "description": "The duration of the video in seconds",
                    "default": 5,
                },
                "stabilityai_config": {
                    "type": "object",
                    "properties": STABILITYAI_PARAMS_CONFIG["text_to_video"],
                    "description": "Config to use when stabilityai engine is used",
                },
                "fal_config": {
                    "type": "object",
                    "properties": FAL_VIDEO_GEN_PARAMS_CONFIG["text_to_video"],
                    "description": "Config to use when fal engine is used",
                },
                "ark_config": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Ark endpoint ID for video generation (e.g. Seedance 2.0 endpoint). If omitted, falls back to ARK_VIDEO_MODEL env.",
                        },
                        "resolution": {"type": "string"},
                        "ratio": {"type": "string"},
                        "seed": {"type": "integer"},
                        "camera_fixed": {"type": "boolean"},
                        "watermark": {"type": "boolean"},
                        "generate_audio": {"type": "boolean"},
                        "draft": {"type": "boolean"},
                        "return_last_frame": {"type": "boolean"},
                        "service_tier": {"type": "string"},
                        "execution_expires_after": {"type": "integer"},
                        "callback_url": {"type": "string"},
                    },
                    "description": "Config to use when ark engine is used",
                },
            },
            "required": ["prompt", "name"],
        },
        "image_to_video": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "The ID of the image in VideoDB to use for video generation",
                },
                "name": {
                    "type": "string",
                    "description": "Description of the video generation run",
                },
                "prompt": {
                    "type": "string",
                    "description": "Text prompt to guide the video generation",
                },
                "duration": {
                    "type": "number",
                    "description": "The duration of the video in seconds",
                    "default": 5,
                },
                "fal_config": {
                    "type": "object",
                    "properties": FAL_VIDEO_GEN_PARAMS_CONFIG["image_to_video"],
                    "description": "Config to use when fal engine is used",
                },
                "ark_config": {
                    "type": "object",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "Ark endpoint ID for video generation (e.g. Seedance 2.0 endpoint). If omitted, falls back to ARK_VIDEO_MODEL env.",
                        },
                        "resolution": {"type": "string"},
                        "ratio": {"type": "string"},
                        "seed": {"type": "integer"},
                        "camera_fixed": {"type": "boolean"},
                        "watermark": {"type": "boolean"},
                        "generate_audio": {"type": "boolean"},
                        "draft": {"type": "boolean"},
                        "return_last_frame": {"type": "boolean"},
                        "service_tier": {"type": "string"},
                        "execution_expires_after": {"type": "integer"},
                        "callback_url": {"type": "string"},
                    },
                    "description": "Config to use when ark engine is used",
                },
            },
            "required": ["prompt", "image_id", "name"],
        },
    },
    "required": ["job_type", "collection_id", "engine"],
}


class VideoGenerationAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "video_generation"
        self.description = "Creates videos using ONE specific model/engine. Only use this agent when the request mentions exactly ONE model/engine, without any comparison words like 'compare', 'test', 'versus', 'vs' and no connecting words (and/&/,) between model names. If the request mentions wanting to compare models or try multiple engines, do not use this agent - use the comparison agent instead."
        self.parameters = VIDEO_GENERATION_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str,
        job_type: str,
        engine: str,
        text_to_video: Optional[dict] = None,
        image_to_video: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Generates video using Stability AI's API based on input text prompt.
        :param collection_id: The collection ID to store the generated video
        :param job_type: The type of video generation job to perform
        :param engine: The engine to use for video generation
        :param text_to_video: The text to convert to video
        :param image_to_video: The image to convert to video
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Response containing the generated video ID
        """
        try:
            media = None
            self.videodb_tool = VideoDBTool(collection_id=collection_id)
            stealth_mode = kwargs.get("stealth_mode", False)
            run_dir = self.session.state.get("run_dir")
            if not isinstance(run_dir, str) or not run_dir:
                raise Exception("run_dir not initialized")

            if engine not in SUPPORTED_ENGINES:
                raise Exception(f"{engine} not supported")

            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Processing...",
            )
            if not stealth_mode:
                self.output_message.content.append(video_content)

            if engine == "stabilityai":
                STABILITYAI_API_KEY = os.getenv("STABILITYAI_API_KEY")
                if not STABILITYAI_API_KEY:
                    raise Exception("Stability AI API key not found")
                video_gen_tool = StabilityAITool(api_key=STABILITYAI_API_KEY)
                config_key = "stabilityai_config"
            elif engine == "fal":
                FAL_KEY = os.getenv("FAL_KEY")
                if not FAL_KEY:
                    raise Exception("FAL API key not found")
                video_gen_tool = FalVideoGenerationTool(api_key=FAL_KEY)
                config_key = "fal_config"

            elif engine == "videodb":
                video_gen_tool = VDBVideoGenerationTool()
                config_key = "videodb_config"
            elif engine == "ark":
                ARK_API_KEY = os.getenv("ARK_API_KEY")
                if not ARK_API_KEY:
                    ark_key, ark_model = _load_ark_from_llm_env()
                    if ark_key:
                        ARK_API_KEY = ark_key
                    if ark_model and not os.getenv("ARK_VIDEO_MODEL") and not os.getenv("ARK_SEEDANCE_MODEL"):
                        os.environ["ARK_VIDEO_MODEL"] = ark_model
                if not ARK_API_KEY:
                    raise Exception("ARK API key not found")
                ARK_BASE_URL = os.getenv(
                    "ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
                )
                video_gen_tool = ArkVideoGenerationTool(
                    api_key=ARK_API_KEY, base_url=ARK_BASE_URL
                )
                config_key = "ark_config"
            else:
                raise Exception(f"{engine} not supported")

            output_file_name = f"video_{job_type}_{str(uuid.uuid4())}.mp4"
            output_path = os.path.join(run_dir, "results", output_file_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if not os.access(os.path.dirname(output_path), os.W_OK):
                raise PermissionError(
                    f"No write permission for output directory: {os.path.dirname(output_path)}"
                )

            if job_type == "text_to_video":
                prompt = text_to_video.get("prompt")
                video_name = text_to_video.get("name")
                duration = text_to_video.get("duration", 5)
                config = text_to_video.get(config_key, {})
                if engine == "ark" and isinstance(config, dict):
                    model_val = config.get("model")
                    if not isinstance(model_val, str) or not model_val.startswith("ep-"):
                        config["model"] = os.getenv("ARK_VIDEO_MODEL") or os.getenv("ARK_SEEDANCE_MODEL")
                if prompt is None:
                    raise Exception("Prompt is required for video generation")
                if engine == "ark" and isinstance(duration, (int, float)) and duration > 15:
                    max_seg = 15
                    seg_count = int(math.ceil(float(duration) / float(max_seg)))
                    seg_durations = [max_seg] * seg_count
                    total = sum(seg_durations)
                    if total != int(duration):
                        seg_durations[-1] = int(duration) - sum(seg_durations[:-1])
                    uploaded_segments = []
                    for i, seg_d in enumerate(seg_durations, start=1):
                        seg_prompt = (
                            f"{prompt}\n"
                            f"这是30秒一镜到底的第{i}/{seg_count}段（每段不超过15秒）。"
                            "镜头连续不中断、不跳切、不换景；环境、人物、冰箱外观保持一致。"
                            "本段开头必须承接上一段结尾画面继续；本段结尾为下一段留出自然衔接姿势。"
                        )
                        seg_output = os.path.join(
                            run_dir,
                            "results",
                            f"video_{job_type}_{uuid.uuid4().hex}_seg{i}.mp4",
                        )
                        self.output_message.actions.append(
                            f"Generating segment {i}/{seg_count} using <b>{engine}</b> (duration {seg_d}s)"
                        )
                        self.output_message.push_update()
                        video_gen_tool.text_to_video(
                            prompt=seg_prompt,
                            save_at=seg_output,
                            duration=seg_d,
                            config=config,
                        )
                        self.output_message.actions.append("Uploading segment to VideoDB")
                        self.output_message.push_update()
                        uploaded = self.videodb_tool.upload(
                            seg_output,
                            source_type="file_path",
                            media_type="video",
                            name=f"{video_name} [{i}/{seg_count}]",
                        )
                        uploaded_segments.append(uploaded)
                    self.output_message.actions.append("Concatenating segments in VideoDB")
                    self.output_message.push_update()
                    stream_url = self.videodb_tool.concat_videos([s["id"] for s in uploaded_segments])
                    media = {
                        "id": uploaded_segments[0]["id"],
                        "collection_id": uploaded_segments[0]["collection_id"],
                        "stream_url": stream_url,
                        "name": video_name,
                    }
                else:
                    self.output_message.actions.append(
                        f"Generating video using <b>{engine}</b> for prompt <i>{prompt}</i>"
                    )
                    self.output_message.push_update()
                    response = video_gen_tool.text_to_video(
                        prompt=prompt,
                        save_at=output_path,
                        duration=duration,
                        config=config,
                    )

                    if response:
                        media = response
            elif job_type == "image_to_video":
                image_id = image_to_video.get("image_id")
                video_name = image_to_video.get("name")
                duration = image_to_video.get("duration", 5)
                config = image_to_video.get(config_key, {})
                prompt = image_to_video.get("prompt")
                if engine == "ark" and isinstance(config, dict):
                    model_val = config.get("model")
                    if not isinstance(model_val, str) or not model_val.startswith("ep-"):
                        config["model"] = os.getenv("ARK_VIDEO_MODEL") or os.getenv("ARK_SEEDANCE_MODEL")

                # Validate duration bounds
                if (
                    not isinstance(duration, (int, float))
                    or duration <= 0
                    or duration > 60
                ):
                    raise ValueError(
                        "Duration must be a positive number between 1 and 60 seconds"
                    )

                if not image_id:
                    raise ValueError(
                        "Missing required parameter: 'image_id' for image-to-video generation"
                    )

                image_data = self.videodb_tool.get_image(image_id)
                if not image_data:
                    raise ValueError(
                        f"Image with ID '{image_id}' not found in collection "
                        f"'{collection_id}'. Please verify the image ID."
                    )

                image_url = None
                if isinstance(image_data.get("url"), str) and image_data.get("url"):
                    image_url = image_data["url"]
                else:
                    image_url = self.videodb_tool.generate_image_url(image_id)

                if not image_url:
                    raise ValueError(
                        f"Image with ID '{image_id}' exists but has no "
                        f"associated URL. This might indicate data corruption."
                    )

                self.output_message.actions.append(
                    f"Generating video using <b>{engine}</b> for <a href='{image_url}'>url</a>"
                )
                if not stealth_mode:
                    self.output_message.push_update()

                if hasattr(video_gen_tool, "image_to_video"):
                    video_gen_tool.image_to_video(
                        image_url=image_url,
                        save_at=output_path,
                        duration=duration,
                        config=config,
                        prompt=prompt,
                    )
                else:
                    raise Exception(f"{engine} does not support image_to_video")
            else:
                raise Exception(f"{job_type} not supported")

            if media is None:
                self.output_message.actions.append(
                    f"Generated video saved at <i>{output_path}</i>"
                )
                self.output_message.push_update()
                media = {}

            if not isinstance(media, dict) or not media.get("stream_url"):
                self.output_message.actions.append("Uploading generated video to VideoDB")
                self.output_message.push_update()
                uploaded = self.videodb_tool.upload(
                    output_path,
                    source_type="file_path",
                    media_type="video",
                    name=video_name,
                )
                media = uploaded
                self.output_message.actions.append(
                    f"Uploaded generated video to VideoDB with Video ID {media['id']}"
                )

            stream_url = media["stream_url"]
            id = media["id"]
            collection_id = media["collection_id"]
            name = media["name"]
            meta_path = os.path.join(
                run_dir, "script", f"video_generation_{id or uuid.uuid4().hex}.json"
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "engine": engine,
                        "job_type": job_type,
                        "output_path": output_path,
                        "media": media,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            video_content.video = VideoData(
                stream_url=stream_url,
                id=id,
                collection_id=collection_id,
                name=name,
            )
            video_content.status = MsgStatus.success
            video_content.status_message = "Here is your generated video"
            self.output_message.push_update()
            self.output_message.publish()

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent: {e}")
            video_content.status = MsgStatus.error
            video_content.status_message = "Failed to generate video"
            self.output_message.push_update()
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Generated video ID {media['id']}",
            data={
                "video_id": media["id"],
                "video_stream_url": stream_url,
                "video_content": video_content,
            },
        )


def _load_ark_from_llm_env() -> tuple[Optional[str], Optional[str]]:
    candidates = []
    env_path = os.getenv("LLM_ENV_MD")
    if env_path:
        candidates.append(env_path)
    candidates.append("/Users/bytedance/Desktop/Workspace/code/LLM_env.md")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(str(parent / "LLM_env.md"))
    for p in candidates:
        if not p:
            continue
        try:
            text = Path(p).read_text(encoding="utf-8")
        except Exception:
            continue
        m_key = re.search(r"API Key:\s*([^\s]+)", text)
        m_model = re.search(r"doubao-seedance-2-0-260128\)\s*\nEndpoint:\s*(ep-[\w-]+)", text)
        key = m_key.group(1) if m_key else None
        model = m_model.group(1) if m_model else None
        if key or model:
            return key, model
    return None, None
