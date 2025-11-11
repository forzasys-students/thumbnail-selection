import requests
import pandas as pd

# CONFIG
TEAM_NAME = "IF Elfsborg"     # e.g. "GAIS", "IFK Göteborg", "Hammarby IF"
EVENT_TYPE = "goal"           # e.g. "goal", "corner", "yellow card"
OUTPUT_FILE = f"allsvenskan_{TEAM_NAME.replace(' ', '_')}_{EVENT_TYPE}.csv"

BASE_URL = "https://api.forzify.com/allsvenskan/event"
COUNT = 100
MAX_PAGES = 50

# FETCH + FILTER
all_records = []

for i in range(MAX_PAGES):
    offset = i * COUNT
    params = {"count": COUNT, "from": offset}
    print(f"Fetching batch {i+1} (from={offset}) ...")

    r = requests.get(BASE_URL, params=params)
    if r.status_code != 200:
        print(f"Request failed with status {r.status_code}")
        break

    events = r.json().get("events", [])
    if not events:
        print("No more events found.")
        break

    for e in events:
        tag = e.get("tag", {})
        playlist = e.get("playlist", {})
        game = playlist.get("game", {})
        home_team = game.get("home_team", {}).get("name", "")
        away_team = game.get("visiting_team", {}).get("name", "")
        action = tag.get("action", "")
        video_url = playlist.get("video_url", "")

        if not video_url:
            continue

        # Check if the event involves your target team
        if (TEAM_NAME.lower() in home_team.lower() or TEAM_NAME.lower() in away_team.lower()) \
            and EVENT_TYPE.lower() in action.lower():
            all_records.append({
                "event_type": action,
                "team_home": home_team,
                "team_away": away_team,
                "game_id": game.get("id"),
                "video_url": video_url,
                "score": e.get("score"),
                "duration_ms": playlist.get("duration_ms"),
                "phase": game.get("phase"),
                "tournament_name": game.get("tournament_name"),
                "recording_timestamp": playlist.get("recording_timestamp"),
                "frontend_url": playlist.get("frontend_url"),
                "thumbnail_url": playlist.get("thumbnail_url"),
            })

# SAVE RESULTS
if all_records:
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Saved {len(df)} '{EVENT_TYPE}' events involving '{TEAM_NAME}' to '{OUTPUT_FILE}'")
else:
    print("\n No events found matching filters.")
