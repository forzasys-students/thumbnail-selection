import json
from typing import Dict, List, Tuple

def parse_camera_labels(label_file: str) -> Dict[int, List[Tuple[float, float, str]]]:
    """
    Parse SoccerNet Labels-cameras.json.

    SoccerNet provides camera annotations as a list of events:
        - "gameTime": human-readable half + mm:ss (e.g. "1 - 01:02")
        - "position": timestamp in milliseconds relative to the half video
        - "label": camera type (e.g. "Main camera center", "Close-up player")

    Each annotation marks the end of its label’s segment.
    """

    # Load JSON file
    with open(label_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    annotations = data.get("annotations", []) or []

    # Organize annotations per half (1st or 2nd half)
    annotations_by_half: Dict[int, List[Tuple[float, str]]] = {1: [], 2: []}

    # Parse raw annotations
    for annotation in annotations:
        game_time = annotation.get("gameTime")     # e.g. "1 - 01:02"
        label = (annotation.get("label") or "unknown").strip()
        position = annotation.get("position")      # milliseconds as string

        if game_time is None or position is None:
            continue  # skip incomplete entries

        try:
            half_number = int(game_time.split(" - ")[0].strip())     # "1 - ..." → 1
            time_seconds = float(int(str(position))) / 1000.0        # ms → s
        except Exception:
            continue  # skip malformed annotation

        if half_number in (1, 2):
            annotations_by_half[half_number].append((time_seconds, label))

    # This will store the final shot segments
    shots: Dict[int, List[Tuple[float, float, str]]] = {1: [], 2: []}

    # Build segments per half
    for half_number in (1, 2):
        cuts = sorted(annotations_by_half[half_number], key=lambda x: x[0])
        if not cuts:
            continue  # no annotations for this half

        # If the first annotation does not start at time 0,
        # add a segment from 0 → first_cut with that first label
        first_time, first_label = cuts[0]
        if first_time > 0.0:
            shots[half_number].append((0.0, first_time, first_label))

        # Each later annotation closes a segment
        # Example: [prev_time, current_time) → label of current annotation
        for i in range(1, len(cuts)):
            previous_time = cuts[i - 1][0]
            current_time, current_label = cuts[i]
            if current_time > previous_time:
                shots[half_number].append((previous_time, current_time, current_label))

        # Extend the last annotation by +60 seconds,
        # so it covers something instead of ending immediately.
        # The frame extractor will clamp this to the actual video duration.
        last_time, last_label = cuts[-1]
        shots[half_number].append((last_time, last_time + 60.0, last_label))

    return shots
