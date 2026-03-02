import os
import sys

# Setup environment variables using the keys from your maas_env.txt
os.environ["ARK_API_KEY"] = "31a38e5c-8245-40ba-acf5-145bdf4be9ad"
os.environ["VOLCENGINE_TTS_APPID"] = "3570190948"
os.environ["VOLCENGINE_TTS_ACCESS_TOKEN"] = "PYjH2Ff5UjvJI8sWtT3uAOpdpHAmFxfb"
# The default LLM provider mapped to Volcano
os.environ["DEFAULT_LLM"] = "volcengine"

# Ensure director package can be found
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from director.core.session import Session, InputMessage
from director.agents.text_to_movie import TextToMovieAgent

def main():
    print("🎬 Starting Volcengine TextToMovie Agent Test...")
    
    # Needs a pseudo session
    from director.db.base import BaseDB
    class MockDB(BaseDB):
        def __init__(self):
            pass
        def get_context_messages(self, session_id):
            return {"reasoning": []}
        def add_or_update_msg_to_conv(self, **kwargs):
            pass
        def add_or_update_context_msg(self, session_id, context):
            pass
        def create_session(self, **kwargs):
            pass
        def get_conversations(self, session_id):
            return []
        def get_session(self, session_id):
            return {}
        def get_sessions(self):
            return []
        def health_check(self):
            return True

    session = Session(session_id="test_volcengine_session", db=MockDB())
    
    # Storyline to test
    storyline = "A futuristic cyberpunk city raining neon lights. A lone samurai stands on a rooftop looking at flying cars."
    
    # Constructing arguments for the agent mimicking what Reasoning Engine passes
    agent_args = {
        "collection_id": "test_collection",
        "engine": "volcengine",        # Specifies Seedance for Video
        "audio_engine": "volcengine",  # Specifies Volcengine TTS for Audio
        "job_type": "text_to_movie",
        "text_to_movie": {
            "storyline": storyline,
            "sound_effects_description": "A dark, intense cyberpunk electronic synth music playing in the background.",
            "video_volcengine_config": {
                "model": "ep-20260225233757-g682r", # Seedance
                "ratio": "16:9"
            },
            "audio_volcengine_config": {
                "voice_type": "BV001_streaming", # Typical TTS streaming voice
            }
        }
    }

    # Initialize agent
    agent = TextToMovieAgent(session=session)
    
    print("\n[+] Triggering GenAI with Volcengine...")
    try:
        response = agent.run(**agent_args)
        print("\n✅ Task Complete!")
        print("Response:", response)
    except Exception as e:
        print("\n❌ Task Failed!")
        print("Error details:", str(e))

if __name__ == "__main__":
    main()
