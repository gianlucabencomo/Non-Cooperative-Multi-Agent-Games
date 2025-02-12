import cv2

def create_video_writer(output_file: str = "./output.mp4", width: int = 1920, height: int = 1080, fps: int = 60):
    """Create a video writer object to save frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    return cv2.VideoWriter(output_file, fourcc, fps, (width, height))

def load_keyframes(filepath):
    """Load keyframes from a txt file."""
    with open(filepath, "r") as f:
        keyframes = [list(map(float, line.strip().split())) for line in f.readlines()]
    return keyframes

def show_video(frames: list, fps: int = 60):
    """Display the recorded frames in an OpenCV window."""
    for frame in frames:
        cv2.imshow("Simulation", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):  # Press 'q' to quit
            break
    cv2.destroyAllWindows()