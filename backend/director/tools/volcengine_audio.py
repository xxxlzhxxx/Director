import os
import json
import uuid
import requests
import base64

PARAMS_CONFIG = {
    "text_to_speech": {
        "voice_type": {
            "type": "string",
            "description": "Voice type for the text-to-speech generation. e.g. BV001_streaming",
            "default": "BV001_streaming",
        },
        "encoding": {
            "type": "string",
            "description": "Encoding format. mp3, wav, pcm",
            "default": "mp3",
        },
        "speed_ratio": {
            "type": "number",
            "description": "Speed ratio of the speaking voice",
            "default": 1.0,
        },
        "volume_ratio": {
            "type": "number",
            "description": "Volume ratio of the speaking voice",
            "default": 1.0,
        },
        "pitch_ratio": {
            "type": "number",
            "description": "Pitch ratio of the speaking voice",
            "default": 1.0,
        }
    },
}

class VolcengineAudioTool:
    def __init__(self, app_id: str = None, access_token: str = None):
        self.app_id = app_id or os.environ.get("VOLCENGINE_TTS_APPID")
        self.access_token = access_token or os.environ.get("VOLCENGINE_TTS_ACCESS_TOKEN")
        
        if not self.app_id or not self.access_token:
            raise Exception("Volcengine TTS credentials not found. Please provide or set VOLCENGINE_TTS_APPID / VOLCENGINE_TTS_ACCESS_TOKEN.")
            
        self.host = "openspeech.bytedance.com"
        self.api_url = f"https://{self.host}/api/v1/tts"
        self.cluster = "volcano_tts"

    def text_to_speech(self, text: str, save_at: str, config: dict):
        """
        Generate TTS using Volcengine API
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        request_json = {
            "app": {
                "appid": self.app_id,
                "token": "access_token", # Required field for structure, but authorization is via Bearer token
                "cluster": self.cluster
            },
            "user": {
                "uid": "director_user"
            },
            "audio": {
                "voice_type": config.get("voice_type", "BV001_streaming"),
                "encoding": config.get("encoding", "mp3"),
                "speed_ratio": config.get("speed_ratio", 1.0),
                "volume_ratio": config.get("volume_ratio", 1.0),
                "pitch_ratio": config.get("pitch_ratio", 1.0)
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson"
            }
        }

        try:
            response = requests.post(self.api_url, json=request_json, headers=headers)
            response.raise_for_status()
            resp_data = response.json()
            
            if "data" in resp_data:
                # The returned data is a base64 encoded string of the audio content
                audio_data = base64.b64decode(resp_data["data"])
                with open(save_at, "wb") as f:
                    f.write(audio_data)
                return True
            else:
                raise Exception(f"Failed to generate TTS: {resp_data.get('message', 'Unknown Error')}")
                
        except Exception as e:
            raise Exception(f"Error generating Volcengine TTS: {str(e)}")

    def generate_sound_effect(self, prompt: str, save_at: str, duration: float, config: dict):
        """
        Volcengine TTS does not do native free-form sound effects from text out of the box like ElevenLabs does.
        If SFX generation is needed by the text_to_movie agent for background music, we will log a warning 
        and fallback to generating a spoken audio or returning None. 
        For true SFX or BGM, an Audio generation model is needed.
        """
        print("[!] Volcengine Audio Tool called for SFX. Returning None to skip or fallback to VideoDB.")
        return None
