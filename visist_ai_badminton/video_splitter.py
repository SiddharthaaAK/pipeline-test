import sys
import os
import cv2

def split_video(video_path, output_dir, segment_duration=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_frames = int(segment_duration * fps)

    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    
    segment_idx = 0
    frame_idx = 0

    while frame_idx < total_frames:
        output_path = os.path.join(output_dir, f"{name}_segment{segment_idx}{ext}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        for _ in range(segment_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_idx += 1
        
        out.release()
        segment_idx += 1
    
    cap.release()
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 video_splitter.py <video_path> <output_dir>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)
    status = split_video(video_path, output_dir)
    sys.exit(status)
