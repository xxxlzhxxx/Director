import os
import sys
import base64
import time

# Configure Volcengine endpoints
os.environ["ARK_API_KEY"] = "31a38e5c-8245-40ba-acf5-145bdf4be9ad"

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from director.tools.volcengine_tool import VolcengineArkTool

# To combine them properly into 30s without waiting 6 minutes for videoDB, we can use ffmpeg directly locally, or videodb. 
# We'll generate 6 distinct 5s videos (3 for cat, 3 for pig) using Seedance 2.0.
# Then rely on VideoDB (if configured) or just local python list to represent them.

def encode_image_base64(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    if filepath.lower().endswith('.png'):
        return f"data:image/png;base64,{encoded_string}"
    else:
        return f"data:image/jpeg;base64,{encoded_string}"

def main():
    print("🎬 Testing 30-Second Video Generation Pipeline using Cats and Pigs...")
    tool = VolcengineArkTool()
    
    cat_image = "/Users/bytedance/Workspace/resources/pic/cat1.jpeg"
    pig_image = "/Users/bytedance/Workspace/resources/pic/pig1.jpg"
    
    cat_url = encode_image_base64(cat_image)
    pig_url = encode_image_base64(pig_image)
    
    # 3 Scenes x 10 seconds = 30 seconds
    storyboard = [
        {"img": cat_url, "prompt": "A cute cat cautiously steps into a magical sunlit grassy field, looking around in wonder."},
        {"img": pig_url, "prompt": "A small pink pig happily trotting towards the camera on a dirt path in the same sunny field."},
        {"img": cat_url, "prompt": "The cute cat chasing a beautiful glowing butterfly over the green grass, dynamic action shot."}
    ]
    
    generated_videos = []
    
    for i, scene in enumerate(storyboard):
        save_at = os.path.join(os.path.dirname(__file__), f"backend/director/downloads/scene_{i+1}_10s.mp4")
        print(f"\n[*] Generating Scene {i+1}/3 (10s) ...")
        print(f"Prompt: {scene['prompt']}")
        try:
            tool.text_to_video(
                prompt=scene["prompt"],
                save_at=save_at,
                duration=10,  # 10s per API limit (4-15s allowed)
                config={"model": "ep-20260225233757-g682r", "ratio": "16:9"},
                image_url=scene["img"]
            )
            generated_videos.append(save_at)
            print(f"✅ Scene {i+1} saved to: {save_at}")
        except Exception as e:
            print(f"❌ Scene {i+1} failed: {e}")
            break

    print("\n🏁 Generation Phase Complete!")
    print("Next step in Director: Send these 6 videos to VideoDB or FFmpeg to concatenate into a single 30s stream.")
    print("Generated clips:", generated_videos)

if __name__ == "__main__":
    main()
