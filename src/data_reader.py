# locate game/video paths

import os

def get_video_path(data_path, league, season, game, half="1_224p.mkv"):
    """
    Construct the path to a specific game video.
    Example usage:
        get_video_path(
            "C:/Users/roshi/Desktop/MasterOppgave/data/SoccerNet",
            "england_epl",
            "2014-2015",
            "2014-08-16 - 12-45 - Manchester United 1-2 Swansea"
        )
    """
    return os.path.join(data_path, league, season, game, half)
