import cv2
import tempfile
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Sightengine API credentials
API_USER = os.getenv("API_USER")
API_SECRET = os.getenv("API_SECRET")

def extract_frames(video_path, frame_rate=1):
    """
    Extract frames from the video at the specified frame rate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if fps > 0 else 1

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            try:
                frames.append(frame)  # Keep as BGR for temporary file saving
            except cv2.error as e:
                print(f"Error converting frame {frame_count}: {e}")
        frame_count += 1

    cap.release()
    if not frames:
        raise ValueError("No valid frames extracted. Check the video file or frame rate.")
    return frames

def classify_frame_with_api(frame):
    """
    Classify a single frame using the Sightengine API.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            cv2.imwrite(temp_image.name, frame)
            temp_image_path = temp_image.name

        url = "https://api.sightengine.com/1.0/check.json"
        files = {'media': open(temp_image_path, 'rb')}
        data = {
            'models': 'nudity',
            'api_user': API_USER,
            'api_secret': API_SECRET
        }
        print(f"Sending frame to API: {temp_image_path}")

        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        print(f"API Response: {result}")

        if "nudity" in result:
            return result["nudity"]["raw"]
        else:
            print(f"Unexpected response: {result}")
            return 0.0
    except requests.exceptions.RequestException as e:
        print(f"Error sending frame to Sightengine API: {e}")
        return 0.0
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def analyze_video(uploaded_video):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name

    frames = extract_frames(video_path, frame_rate=1)

    if not frames:
        raise ValueError("No frames were extracted from the video. Please upload a valid video file.")

    nsfw_scores = []
    for idx, frame in enumerate(frames):
        try:
            score = classify_frame_with_api(frame)
            nsfw_scores.append(score)
            print(f"Frame {idx}: NSFW Score = {score:.2f}")
        except Exception as e:
            print(f"Error processing frame {idx}: {e}")

    if not nsfw_scores:
        raise ValueError("No scores were calculated. Please check the video and API response.")

    aggregated_score = np.mean(nsfw_scores)
    print(f"Final Aggregated NSFW Score: {aggregated_score:.2f}")
    return aggregated_score

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    try:
        print(f"Processing video: {video_path}")
        frames = extract_frames(video_path, frame_rate=1)
        scores = [classify_frame_with_api(frame) for frame in frames]
        aggregated_score = np.mean(scores)
        print(f"Aggregated NSFW Score: {aggregated_score:.2f}")
    except Exception as e:
        print(f"Error: {e}")
