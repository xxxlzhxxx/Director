import os
import sys
import base64

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
    # Use proper image type
    if filepath.endswith('.png'):
        return f"data:image/png;base64,{encoded_string}"
    else:
        return f"data:image/jpeg;base64,{encoded_string}"

def main():
    print("🎬 Testing VolcengineArkTool Image-to-Video generation directly...")
    
    # 1. Initialize Volcengine tool
    tool = VolcengineArkTool()
    
    # 2. Get an image
    image_path = "/Users/bytedance/Workspace/resources/pic/pig1.jpg"
    print(f"[*] Reading image: {image_path}")
    image_url = encode_image_base64(image_path)
    
    # 3. Formulate Prompt
    prompt = "A cute little pig running happily in a bright green grassy field on a beautifully sunny day, 4k resolution."
    save_at = os.path.join(os.path.dirname(__file__), "backend/director/downloads/test_pig1_video.mp4")
    
    print(f"[*] Dispatching task to Volcengine Seedance 2.0...")
    tool.text_to_video(
        prompt=prompt,
        save_at=save_at,
        duration=5,
        config={"model": "ep-20260225233757-g682r", "ratio": "16:9"},
        image_url=image_url
    )
    
    print(f"✅ Video generated and saved at: {save_at}")

if __name__ == "__main__":
    main()
