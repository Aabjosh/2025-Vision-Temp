import cv2
import os

def record_video(output_video_path, record_duration=10, fps=30):
    """
    Records a video from the webcam and saves it to the specified path.

    Args:
        output_video_path (str): Path to save the recorded video.
        record_duration (int): Duration of the video recording in seconds.
        fps (int): Frames per second for the video.
    """
    cap = cv2.VideoCapture(1)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Get the video frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Save as MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Recording video... Press 'q' to stop early.")

    frame_count = 0
    max_frames = fps * record_duration

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)  # Write the frame to the video file
        cv2.imshow('Recording Video', frame)  # Display the video while recording

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_video_path}")

def extract_frames(video_path, output_folder, frame_interval=10):
    """
    Extracts every nth frame from a video and saves it as an image.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save the extracted frames.
        frame_interval (int): Interval between frames to save (default: every 10th frame).

    Returns:
        int: Total number of frames saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save every nth frame
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            saved_count += 1

        # Display the frame being processed
        cv2.imshow('Extracting Frames', frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Frame extraction stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Extraction complete. {saved_count} frames saved.")
    return saved_count

if __name__ == "__main__":
    # File paths and parameters
    output_video_path = 'calibration_video.mp4'  # Path to save the recorded video
    output_folder = 'calibration_images'  # Folder to save extracted frames
    record_duration = 30  # Record for 10 seconds
    fps = 100  # Frames per second for the video
    frame_interval = 20  # Save every 10th frame

    # Step 1: Record video
    record_video(output_video_path, record_duration, fps)

    # Step 2: Extract frames
    extract_frames(output_video_path, output_folder, frame_interval)
