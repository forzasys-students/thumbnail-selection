import os
from getpass import getpass
from SoccerNet.Downloader import SoccerNetDownloader, getListGames

def download_data(data_path, split="train", max_games=25):
    """
    Download SoccerNet data for a given split (train/valid/test).

    Downloads for each selected game:
        - 1_224p.mkv (first half video, low-res)
        - 2_224p.mkv (second half video, low-res)
        - Labels-cameras.json (camera annotations)
    """
    
    # Prompt for NDA password (stored locally, not in repo)
    password = getpass("Enter your SoccerNet NDA password: ")
    downloader = SoccerNetDownloader(LocalDirectory=data_path)
    downloader.password = password

    # Get all games in the split and limit to `max_games`
    games = getListGames(split=split)
    selected_games = games[:max_games]

    print(f"Found {len(games)} games in split '{split}'. Downloading {len(selected_games)} of them.")

    for game in selected_games:
        print(f"\n=== Downloading {game} ===")
        downloader.downloadGame(
            game=game,
            files=["1_224p.mkv", "2_224p.mkv", "Labels-cameras.json", "video.ini"]
        )

if __name__ == "__main__":
    data_path = "C:/Users/roshi/Desktop/MasterOppgave/data/SoccerNet"
    os.makedirs(data_path, exist_ok=True)
    download_data(data_path, split="train", max_games=25)
