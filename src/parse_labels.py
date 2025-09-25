import json

def parse_camera_labels(label_file, fps=25.0):
    """
    Parse Labels-cameras.json into per-half shot lists.

    Instead of using the raw 'position' (absolute frame index),
    we parse the 'gameTime' string which looks like "2 - 04:59":
        - First number = half (1 or 2)
        - Time = mm:ss from the start of that half
    We then convert mm:ss → seconds → frame index (using fps=25).

    Returns:
        {
            1: [(frame_idx, label), ...],  # annotations for first half
            2: [(frame_idx, label), ...]   # annotations for second half
        }
    """
    with open(label_file, "r") as f:
        data = json.load(f)

    # Preallocate per-half containers
    shots_by_half = {1: [], 2: []}

    # Loop through all annotations in the file
    for ann in data.get("annotations", []):
        game_time = ann.get("gameTime")
        label = ann.get("label", "unknown")

        if not game_time:
            continue # Skip malformed entries

        try:
            # Example: "2 - 04:59"
            half_str, time_str = game_time.split(" - ")
            half = int(half_str.strip())
            minutes, seconds = map(int, time_str.split(":"))
            total_seconds = minutes * 60 + seconds
            # Convert time → frame index
            frame_idx = int(total_seconds * fps)

            # Store result if half is valid
            if half in shots_by_half:
                shots_by_half[half].append((frame_idx, label))
        except Exception as e:
            # Log and skip if parsing fails
            print(f"Skipping bad annotation {ann}: {e}")
            continue

    return shots_by_half
