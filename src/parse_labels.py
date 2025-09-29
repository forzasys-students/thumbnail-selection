import json

def parse_camera_labels(label_file):
    """
    Parse Labels-cameras.json into per-half shot windows using gameTime only.

    We treat each annotation as the START of a shot. The END is the next
    annotation *in the same half*. If the next annotation is in a different
    half or missing, we set a provisional end (will be clamped by video length).

    Returns:
        {
            1: [(start_sec, end_sec, label), ...],
            2: [(start_sec, end_sec, label), ...],
        }
    """
    with open(label_file, "r") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    # Pre-parse into (half, sec, label) list
    parsed = []
    for ann in anns:
        gt = ann.get("gameTime")
        label = ann.get("label", "unknown")
        if not gt:
            continue
        try:
            half_str, time_str = gt.split(" - ")
            half = int(half_str.strip())
            m, s = map(int, time_str.split(":"))
            sec = m * 60 + s
            parsed.append((half, sec, label))
        except Exception:
            # Skip malformed
            continue

    # Build per-half shot windows: current -> next in same half
    shots_by_half = {1: [], 2: []}
    for i, (h, sec, label) in enumerate(parsed):
        # find next annotation in same half
        end_sec = None
        if i + 1 < len(parsed):
            next_h, next_sec, _ = parsed[i + 1]
            if next_h == h:
                end_sec = next_sec

        # fallback end (will be clamped later by video duration)
        if end_sec is None:
            end_sec = sec + 60.0  # 1 minute fallback

        if h in shots_by_half:
            shots_by_half[h].append((float(sec), float(end_sec), label))

    return shots_by_half
