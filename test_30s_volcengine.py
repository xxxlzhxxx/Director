import os
import sys
import base64
import time

# Configure Volcengine endpoints
os.environ["ARK_API_KEY"] = "31a38e5c-8245-40ba-acf5-145bdf4be9ad"
os.environ["VOLCENGINE_TTS_APPID"] = "3570190948"
os.environ["VOLCENGINE_TTS_ACCESS_TOKEN"] = "PYjH2Ff5UjvJI8sWtT3uAOpdpHAmFxfb"

# Ensure imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from director.tools.volcengine_tool import VolcengineArkTool

def encode_image_base64(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    if filepath.lower().endswith('.png'):
        return f"data:image/png;base64,{encoded_string}"
    else:
        return f"data:image/jpeg;base64,{encoded_string}"

def main():
    print("🎬 Testing VolcengineArkTool 30s Video Generation...")
    tool = VolcengineArkTool()
    
    # We will try passing duration=30. If Seedance 2.0 rejects it, we catch the exception.
    # We'll use cat1.jpeg
    image_path = "/Users/bytedance/Workspace/resources/pic/cat1.jpeg"
    print(f"[*] Reading image: {image_path}")
    image_url = encode_image_base64(image_path)
    
    prompt = "A cute cat exploring a magical forest, enchanted glowing plants, cinematic lighting, 4k. The camera slowly pans across the landscape."
    save_at = os.path.join(os.path.dirname(__file__), "backend/director/downloads/test_30s_cat_video.mp4")
    
    print(f"[*] Dispatching task to Volcengine Seedance 2.0 with duration=30...")
    try:
        tool.text_to_video(
            prompt=prompt,
            save_at=save_at,
            duration=30,
            config={"model": "ep-20260225233757-g682r", "ratio": "16:9"},
            image_url=image_url
        )
        print(f"✅ 30s Video generated successfully and saved at: {save_at}")
    except Exception as e:
        print(f"❌ Failed requesting 30s duration: {e}")
        print("Note: Most standard video generation models (like Seedance) natively support 4s or 5s clips per request.")

if __name__ == "__main__":
    main()
