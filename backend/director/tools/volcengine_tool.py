import os
import time
import requests
from openai import OpenAI
from director.constants import DOWNLOADS_PATH

PARAMS_CONFIG = {
    "text_to_video": {
        "model": {
            "type": "string",
            "description": "Model to use for video generation",
            "default": "ep-20260225233757-g682r"  # Seedance 2.0
        },
        "ratio": {
             "type": "string",
             "enum": ["16:9", "9:16", "1:1"],
             "default": "16:9"
        }
    },
    "text_to_image": {
        "model": {
            "type": "string",
            "description": "Model to use for image generation",
            "default": "ep-20260228143218-zwr4g"  # Seedream 5.0
        },
        "size": {
             "type": "string",
             "enum": ["2K", "1K", "4K"],
             "default": "2K"
        }
    }
}

class VolcengineArkTool:
    def __init__(self, api_key: str = None):
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            raise ImportError("Please run: pip install volcenginesdkarkruntime")
        
        self.api_key = api_key or os.environ.get("ARK_API_KEY")
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=self.api_key
        )
        # Using OpenAI client strictly for Image generation right now based on standard compatibility
        self.openai_client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3", 
            api_key=self.api_key, 
        )
        self.polling_interval = 5

    def _download_file(self, url: str, save_at: str):
        os.makedirs(os.path.dirname(save_at), exist_ok=True)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_at, "wb") as file:
            file.write(response.content)

    def text_to_video(self, prompt: str, save_at: str, duration: float, config: dict, image_url: str = None):
        """Generate Video using Doubao Seedance 2.0"""
        model = config.get("model", "ep-20260225233757-g682r")
        ratio = config.get("ratio", "16:9")
        
        content = []
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
            
        content.append({"type": "text", "text": prompt})
        
        create_result = self.client.content_generation.tasks.create(
            model=model,
            content=content,
            ratio=ratio,
            duration=int(duration),
            watermark=False,
        )
        
        task_id = create_result.id
        print(f"[*] Started Volcengine Video Task: {task_id}")
        
        while True:
            get_result = self.client.content_generation.tasks.get(task_id=task_id)
            status = get_result.status
            
            if status == "succeeded":
                content_obj = getattr(get_result, 'content', None)
                if not content_obj and hasattr(get_result, 'dict'):
                    content_obj = get_result.dict().get('content')
                elif isinstance(get_result, dict):
                    content_obj = get_result.get('content')

                video_url = getattr(content_obj, 'video_url', None) or (content_obj.get('video_url') if isinstance(content_obj, dict) else None)
                
                if not video_url:
                    raise Exception("Success but no video_url found in the response body.")

                print(f"[*] Downloading Volcengine Video from: {video_url}")
                self._download_file(video_url, save_at)
                break
                
            elif status == "failed":
                raise Exception(f"Video generation failed: {get_result.error}")
                
            else:
                time.sleep(self.polling_interval)

    def text_to_image(self, prompt: str, save_at: str, config: dict):
        """Generate Image using Doubao Seedream 5.0"""
        model = config.get("model", "ep-20260228143218-zwr4g")
        size = config.get("size", "2K")

        imagesResponse = self.openai_client.images.generate( 
            model=model, 
            prompt=prompt,
            size=size,
            response_format="url",
            extra_body={
                "watermark": config.get("watermark", False),
            },
        ) 
        
        image_url = imagesResponse.data[0].url
        if not image_url:
             raise Exception("Success but no image_url found in the response body.")

        print(f"[*] Downloading Volcengine Image from: {image_url}")
        self._download_file(image_url, save_at)
        return {"url": image_url}
