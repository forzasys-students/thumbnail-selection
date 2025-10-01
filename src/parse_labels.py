import json

def parse_camera_labels(label_file):
    """
    Parse SoccerNet camera label annotations into structured shot segments.

    The annotation file `Labels-cameras.json` contains a list of timestamped events,
    where each event specifies:
        - gameTime (e.g., "1 - 12:34" = first half, 12 minutes 34 seconds)
        - label    (e.g., "Main camera center", "Close-up player or field referee")

    This function converts those annotations into per-half "shots":
        - Each annotation marks the START of a shot.
        - The END of a shot is defined by the next annotation in the same half.
        - If no next annotation is found in the same half, we assign a fallback duration
          of +60 seconds (later clamped by video length).

    Returns:
        dict:
            {
                1: [(start_sec, end_sec, label), ...],   # Shots for first half
                2: [(start_sec, end_sec, label), ...],   # Shots for second half
            }
    """
    
    # --- Load JSON annotations ---
    with open(label_file, "r") as f:
        data = json.load(f)

    anns = data.get("annotations", [])

    # --- Pre-parse annotations into tuples (half, second, label) ---
    parsed = []
    for ann in anns:
        gt = ann.get("gameTime")
        label = ann.get("label", "unknown")  # default label if missing
        
        if not gt:
            continue  # skip if no gameTime is present
        
        try:
            # Example format: "1 - 12:34"
            half_str, time_str = gt.split(" - ")
            half = int(half_str.strip())  # which half (1 or 2)
            
            # Convert "mm:ss" → seconds
            m, s = map(int, time_str.split(":"))
            sec = m * 60 + s
            
            parsed.append((half, sec, label))
        except Exception:
            # Skip malformed entries (bad formatting, missing fields)
            continue

    # --- Group into per-half shots ---
    shots_by_half = {1: [], 2: []}

    for i, (h, sec, label) in enumerate(parsed):
        # Default: no explicit end time yet
        end_sec = None
        
        # Look at the next annotation: if it's in the same half, that’s the shot’s end
        if i + 1 < len(parsed):
            next_h, next_sec, _ = parsed[i + 1]
            if next_h == h:
                end_sec = next_sec

        # If no valid "next" annotation, fallback = +60s window
        if end_sec is None:
            end_sec = sec + 60.0  # 1 minute fallback

        # Save shot window (start, end, label) under correct half
        if h in shots_by_half:
            shots_by_half[h].append((float(sec), float(end_sec), label))

    return shots_by_half
