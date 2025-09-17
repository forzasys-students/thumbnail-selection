import os
from getpass import getpass
from SoccerNet.Downloader import SoccerNetDownloader
from frame_extractor import extract_frames

def find_first_video(data_path):
    """Walk the SoccerNet folder and return the first 224p video found."""
    for root, dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith("1_224p.mkv"):  # first half, low-res
                return os.path.join(root, f)
    return None

def run_pipeline():
    # Base paths
    data_path = "C:/Users/roshi/Desktop/MasterOppgave/data/SoccerNet"
    thumb_root = "C:/Users/roshi/Desktop/MasterOppgave/data/thumbnails"

    # Check if the SoccerNet folder is empty
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        print("SoccerNet folder is empty. Downloading videos...")
        password = getpass("Enter your SoccerNet password: ")
        downloader = SoccerNetDownloader(LocalDirectory=data_path)
        downloader.password = password

        try:
            downloader.downloadGames(files=["1_224p.mkv"], split=["valid"])
        except KeyboardInterrupt:
            print("Download cancelled by user. Proceeding with existing files...")
    else:
        print("SoccerNet folder already has data. Skipping download.")

    # Look for an already-downloaded video
    video_file = find_first_video(data_path)
    if not video_file:
        raise RuntimeError("No 224p video found. Did you download at least one?")

    print(f"Using local video: {video_file}")

    # Thumbnails go into a folder named after the game
    safe_name = os.path.relpath(video_file, data_path).replace("\\", "_").replace("/", "_")
    thumb_out = os.path.join(thumb_root, safe_name)
    os.makedirs(thumb_out, exist_ok=True)

    # Extract frames
    extract_frames(video_file, thumb_out, interval=100)

if __name__ == "__main__":
    run_pipeline()
