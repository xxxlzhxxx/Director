import os
import sys
import base64
import time
import cv2

# Configure Volcengine endpoints
os.environ["ARK_API_KEY"] = "31a38e5c-8245-40ba-acf5-145bdf4be9ad"

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from director.tools.volcengine_tool import VolcengineArkTool

def encode_image_base64_from_path(filepath):
    """Encode an image directly from a file path."""
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    if filepath.lower().endswith('.png'):
        return f"data:image/png;base64,{encoded_string}"
    else:
        return f"data:image/jpeg;base64,{encoded_string}"

def extract_last_frame(video_path, output_image_path):
    """Extracts the last frame of a video and saves it as an image."""
    print(f"[*] Extracting last frame from {video_path}...")
    vidcap = cv2.VideoCapture(video_path)
    
    # Get total frames and set position to the last frame
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 2) # Sometimes the very last frame is blank/corrupted
    
    success, image = vidcap.read()
    if success:
        cv2.imwrite(output_image_path, image)
        print(f"[*] Last frame saved to {output_image_path}")
        return output_image_path
    else:
        raise Exception("Failed to read the last frame of the video.")

def main():
    print("🎬 Testing Infinite Zoom/One-Take 30s Video Generation (Chain Generation)...")
    tool = VolcengineArkTool()
    
    # 1. Starting Image
    starting_image_path = "/Users/bytedance/Workspace/resources/pic/cat1.jpeg"
    current_image_url = encode_image_base64_from_path(starting_image_path)
    
    # 2. Define the continuous narrative (3x 10s = 30s)
    # The key to "one-take" is keeping the prompt extremely contiguous and consistent.
    storyboard = [
        {"prompt": "A cute cat cautiously exploring a magical sunlit grassy field, enchanted glowing plants slightly swaying. Slow cinematic continuous forward tracking shot, establishing the scene."},
        {"prompt": "The camera continues to glide forward smoothly. The cat looks up at a glowing blue butterfly that enters the frame, cinematic continuous tracking shot, matching previous lighting."},
        {"prompt": "The camera keeps moving forward without stopping. The cat follows the butterfly deeper into a softly lit grove of ancient trees, seamless continuation, 4k resolution."}
    ]
    
    generated_videos = []
    
    for i, scene in enumerate(storyboard):
        save_video_at = os.path.join(os.path.dirname(__file__), f"backend/director/downloads/chain_scene_{i+1}_10s.mp4")
        print(f"\n==============================================")
        print(f"[*] Generating Flowing Scene {i+1}/3 (10s)")
        print(f"Prompt: {scene['prompt']}")
        
        try:
            # 3. Generate the 10s clip using the current image reference
            tool.text_to_video(
                prompt=scene["prompt"],
                save_at=save_video_at,
                duration=10, 
                config={"model": "ep-20260225233757-g682r", "ratio": "16:9"},
                # Crucial step: Supply the starting frame
                image_url=current_image_url 
            )
            print(f"✅ Scene {i+1} saved to: {save_video_at}")
            generated_videos.append(save_video_at)
            
            # 4. Extract the VERY LAST frame of the generated video to use as the starting point for the next one!
            if i < len(storyboard) - 1:
                next_frame_path = os.path.join(os.path.dirname(__file__), f"backend/director/downloads/bridge_frame_{i+1}.jpg")
                extract_last_frame(save_video_at, next_frame_path)
                
                # Encode this newly extracted frame for the next iteration
                current_image_url = encode_image_base64_from_path(next_frame_path)
                print(f"[*] Ready to chain to Scene {i+2} using extracted frame.")
                
        except Exception as e:
            print(f"❌ Scene {i+1} failed: {e}")
            break

    print("\n🏁 Chain Generation Phase Complete!")
    print("Generated seamless clip segments:", generated_videos)
    print("These clips now start exactly where the previous one ended!")

if __name__ == "__main__":
    main()
