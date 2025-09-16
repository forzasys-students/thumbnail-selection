import os
from getpass import getpass
from SoccerNet.Downloader import SoccerNetDownloader

def download_video(data_path, game_file="1_224p.mkv", split="train"):
    """
    Download a single video file from SoccerNet.
    Requires NDA password.
    """
    password = getpass("Enter your SoccerNet password: ")
    downloader = SoccerNetDownloader(LocalDirectory=data_path)
    downloader.password = password
    downloader.downloadGames(files=[game_file], split=[split])
    print(f"Downloaded {game_file} into {data_path}")

if __name__ == "__main__":
    data_path = "C:/Users/roshi/Desktop/MasterOppgave/data/SoccerNet"
    download_video(data_path)
