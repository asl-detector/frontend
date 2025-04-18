import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from tqdm import tqdm
import concurrent.futures
import os
import json
import gc
import threading
import argparse

# Create a dictionary to store landmarkers by thread id
landmarker_cache = {}
cache_lock = threading.Lock()


def get_landmarker(model_path):
    """Get or create a HandLandmarker for the current thread"""
    thread_id = threading.get_ident()

    # Check if this thread already has a landmarker
    with cache_lock:
        if thread_id not in landmarker_cache:
            BaseOptions = python.BaseOptions
            HandLandmarker = vision.HandLandmarker
            HandLandmarkerOptions = vision.HandLandmarkerOptions
            VisionRunningMode = vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
            )

            landmarker_cache[thread_id] = HandLandmarker.create_from_options(options)
            print(f"Created new landmarker for thread ID {thread_id}")

    return landmarker_cache[thread_id]


def process_batch(batch_data):
    """Process a batch of frames and extract landmark data"""
    start_idx, frames, timestamps, model_path = batch_data
    landmarks_data = []

    # Get the thread-specific landmarker
    landmarker = get_landmarker(model_path)

    for i, (frame, timestamp_ms) in enumerate(zip(frames, timestamps)):
        frame_idx = start_idx + i
        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect hand landmarks
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Extract landmark data
        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_ms": timestamp_ms / 1000,  # Convert to seconds
            "hands": [],
        }

        # Store data for each detected hand
        for hand_idx, (landmarks, handedness) in enumerate(
            zip(result.hand_landmarks, result.handedness)
        ):
            hand_data = {
                "handedness": handedness[0].category_name,
                "confidence": float(handedness[0].score),
                "landmarks": [],
            }

            # Store all landmark points
            for landmark in landmarks:
                hand_data["landmarks"].append(
                    {
                        "x": float(landmark.x),
                        "y": float(landmark.y),
                        "z": float(landmark.z),
                    }
                )

            frame_data["hands"].append(hand_data)

        landmarks_data.append(frame_data)

    return landmarks_data


def cleanup_resources():
    """Clean up all landmarker instances"""
    print(f"Cleaning up {len(landmarker_cache)} landmarker instances")
    for thread_id, landmarker in landmarker_cache.items():
        del landmarker
    landmarker_cache.clear()
    gc.collect()


def main(video_path, model_path, output_json_path):
    """Process a single video file and extract hand landmarks

    Args:
        video_path (str): Path to the video file
        model_path (str): Path to the MediaPipe hand landmarker model
        output_json_path (str): Path to save the output JSON file
    """
    print(f"Processing video: {video_path}")
    print(f"Output JSON path: {output_json_path}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Determine the number of workers to use
    max_workers = os.cpu_count()
    if max_workers is None:
        max_workers = 1
    workers = max(1, max_workers)
    print(f"Using {workers} workers for parallel processing")

    # Open the video file to get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process video in chunks to conserve memory
    chunk_size = 500
    current_frame = 0

    # Container for all landmark data
    all_landmark_data = []

    # Create a single ThreadPoolExecutor for the entire process
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while current_frame < total_frames:
                # Calculate the end frame for this chunk
                end_frame = min(current_frame + chunk_size, total_frames)
                frames_to_process = end_frame - current_frame

                # Read chunk of frames
                chunk_frames = []
                chunk_timestamps = []

                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

                for _ in range(frames_to_process):
                    success, frame = cap.read()
                    if not success:
                        break
                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)
                    chunk_frames.append(frame)
                    chunk_timestamps.append(timestamp_ms)

                # Split chunk into batches for parallel processing
                batch_size = max(1, len(chunk_frames) // workers)
                batches = []

                for i in range(0, len(chunk_frames), batch_size):
                    end_idx = min(i + batch_size, len(chunk_frames))
                    batch_frames = chunk_frames[i:end_idx]
                    batch_timestamps = chunk_timestamps[i:end_idx]
                    batches.append(
                        (current_frame + i, batch_frames, batch_timestamps, model_path)
                    )

                # Process batches in parallel
                chunk_landmarks = []
                future_to_batch = {
                    executor.submit(process_batch, batch): i
                    for i, batch in enumerate(batches)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        chunk_landmarks.extend(batch_results)
                    except Exception as e:
                        print(f"Error processing batch: {e}")

                # Sort results by frame index
                chunk_landmarks.sort(key=lambda x: x["frame_idx"])

                # Add to the main data collection
                all_landmark_data.extend(chunk_landmarks)

                # Update progress
                pbar.update(len(chunk_landmarks))

                # Move to next chunk
                current_frame = end_frame

                # Clear memory
                del chunk_frames
                del chunk_timestamps
                del chunk_landmarks
                gc.collect()

    # Release video capture
    cap.release()

    # Save all landmark data to a JSON file
    print(
        f"Saving landmark data for {len(all_landmark_data)} frames to {output_json_path}"
    )
    with open(output_json_path, "w") as f:
        json.dump(
            {
                "video_info": {
                    "fps": fps,
                    "total_frames": total_frames,
                    "source": video_path,
                },
                "frames": all_landmark_data,
            },
            f,
            indent=2,
        )

    print("Landmark data saved successfully.")

    # Clean up resources
    cleanup_resources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand landmarks from a video")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument(
        "--output", required=True, help="Path to save the output JSON file"
    )
    parser.add_argument(
        "--model", required=True, help="Path to the MediaPipe hand landmark model"
    )

    args = parser.parse_args()

    main(args.video, args.model, args.output)
