from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
import os
import subprocess
import threading
import queue
import json
import csv
import shutil
from werkzeug.utils import secure_filename
import time
import requests
from datetime import datetime, timedelta
import shutil  

# Add near the top with other config
API = "https://api.fotbollplay.se/allsvenskan/event"

# video_asset_id → event index (built once, refreshed every INDEX_TTL seconds)
VIDEO_INDEX: dict = {}
INDEX_LAST_UPDATED: float | None = None
INDEX_TTL = 300  # seconds (5 min)

# Keep legacy cache dict for the proxy endpoint
METADATA_CACHE = {}
CACHE_DURATION = 300


# SAM3 client imports
from sam3.sam3_client import (
    call_sam3_segmentation,
    check_sam3_service_health,
    compose_thumbnail
)

app = Flask(__name__)

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to inference script
SCRIPT_PATH = os.path.join(BASE_DIR, "run_inference_pipeline.sh")

# Output folder for keyframes
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "inference_output", "keyframes")

# Masks and thumbnails folders
MASKS_FOLDER = os.path.join(BASE_DIR, "data", "inference_output", "masks")
THUMBNAILS_FOLDER = os.path.join(BASE_DIR, "data", "inference_output", "thumbnails")

os.makedirs(MASKS_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)

# Queue for progress updates
progress_queue = queue.Queue()
current_status = {"stage": "", "progress": 0, "message": ""}


def cleanup_data_folder():
    """
    Wipe all previous uploads and inference output before starting a new run.

    - Deletes every video file sitting directly inside  data/
    - Removes data/inference_output/ entirely (frames, predictions,
      segments, keyframes, masks, thumbnails)
    - Recreates the empty masks/ and thumbnails/ dirs so Flask routes
      and the compositor still work immediately after cleanup.
    """
    data_dir      = os.path.join(BASE_DIR, "data")
    inference_dir = os.path.join(data_dir, "inference_output")

    # 1. Remove old uploaded video files from data/ (files only, not subdirs)
    if os.path.isdir(data_dir):
        for entry in os.scandir(data_dir):
            if entry.is_file():
                try:
                    os.remove(entry.path)
                    print(f"[Cleanup] Removed old upload: {entry.name}")
                except Exception as exc:
                    print(f"[Cleanup] Could not remove {entry.name}: {exc}")

    # 2. Remove the entire inference_output tree
    if os.path.isdir(inference_dir):
        shutil.rmtree(inference_dir, ignore_errors=True)
        print("[Cleanup] Removed inference_output/")

    # 3. Recreate the dirs that Flask expects to exist straight away
    os.makedirs(MASKS_FOLDER,      exist_ok=True)
    os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)
    print("[Cleanup] Output directories recreated — ready for new run.")


#def run_inference(model, video_path, fps=5, redundancy_reduction=True):
def run_inference(model, video_path, fps=12, redundancy_reduction=True, visual_threshold=0.90, w_face=0.25, w_emotion=0.15, w_pose=0.15, w_iqa=0.35, logo_p=0.50):

    """Run inference pipeline and emit progress updates"""
    global current_status
    
    try:
        current_status = {"stage": "Starting", "progress": 0, "message": "Initializing pipeline..."}
        progress_queue.put(current_status.copy())
        
        # Use Git Bash on Windows
        bash_executable = r"C:\Program Files\Git\bin\bash.exe"
        
        # Debug output
        print(f"Running: {bash_executable} {SCRIPT_PATH} --model {model} --video {video_path}")
        print(f"Working directory: {BASE_DIR}")
        print(
            f"fps={fps}  redundancy_reduction={redundancy_reduction}  "
            f"visual_threshold={visual_threshold}  logo_p={logo_p}  "
            f"w_face={w_face}  w_emotion={w_emotion}  w_pose={w_pose}  w_iqa={w_iqa}"
        )

        cmd = [
            bash_executable, SCRIPT_PATH,
            "--model", model,
            "--video", video_path,
            "--fps", str(fps),
            "--visual_threshold", str(visual_threshold),
            "--logo_p", str(logo_p),
            "--w_face", str(w_face),
            "--w_emotion", str(w_emotion),
            "--w_pose", str(w_pose),
            "--w_iqa", str(w_iqa),
        ]

        if not redundancy_reduction:
            cmd.append("--no_redundancy")

        # Start the subprocess with UTF-8 encoding
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            cwd=BASE_DIR,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        stage_progress = {
            "[STEP 1]": 25,
            "[STEP 2]": 50,
            "[STEP 3]": 75,
            "[STEP 4]": 90,
            "[DONE]": 100
        }
        
        # Read output line by line
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"Script output: {line}")  # Debug print
                
                # Update current message
                current_status["message"] = line
                
                # Detect stages and update progress
                if "[STEP 1]" in line:
                    current_status["stage"] = "[STEP 1]"
                    current_status["progress"] = 25
                    progress_queue.put(current_status.copy())
                elif "[STEP 2]" in line:
                    current_status["stage"] = "[STEP 2]"
                    current_status["progress"] = 50
                    progress_queue.put(current_status.copy())
                elif "[STEP 3]" in line:
                    current_status["stage"] = "[STEP 3]"
                    current_status["progress"] = 75
                    progress_queue.put(current_status.copy())
                elif "[STEP 4]" in line:
                    current_status["stage"] = "[STEP 4]"
                    current_status["progress"] = 90
                    progress_queue.put(current_status.copy())
                elif "[DONE]" in line:
                    current_status["stage"] = "Complete"
                    current_status["progress"] = 100
                    progress_queue.put(current_status.copy())
                else:
                    # Send general updates periodically
                    progress_queue.put(current_status.copy())
        
        process.wait()
        
        if process.returncode == 0:
            current_status = {
                "stage": "Complete",
                "progress": 100,
                "message": "Pipeline completed successfully!"
            }
        else:
            current_status = {
                "stage": "Error",
                "progress": 0,
                "message": "Pipeline failed with errors"
            }
        progress_queue.put(current_status.copy())
        
    except Exception as e:
        current_status = {
            "stage": "Error",
            "progress": 0,
            "message": f"Error: {str(e)}"
        }
        progress_queue.put(current_status.copy())

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_status

    if request.method == 'GET':
        # Reset stale state from previous run
        current_status = {"stage": "idle", "progress": 0, "message": ""}
        while not progress_queue.empty():
            try:
                progress_queue.get_nowait()
            except Exception:
                break
        return render_template('index.html')

    if request.method == 'POST':
        model = request.form['model']
        video_file = request.files['video']

        if video_file:
            filename = secure_filename(video_file.filename)
            upload_dir = os.path.join(BASE_DIR, "data")
            os.makedirs(upload_dir, exist_ok=True)

            # Reset pipeline state and drain queue before new run
            current_status = {"stage": "Starting", "progress": 0, "message": "Initializing pipeline..."}
            while not progress_queue.empty():
                try:
                    progress_queue.get_nowait()
                except Exception:
                    break

            # Clean previous run's files
            print(f"[Upload] New video received: {filename} — cleaning previous data...")
            cleanup_data_folder()

            video_path = os.path.join(upload_dir, filename)
            video_file.save(video_path)

            fps = int(request.form.get('fps', 12))
            fps = max(1, min(24, fps))
            redundancy_reduction = request.form.get('redundancy_reduction') == '1'

            visual_threshold = float(request.form.get('visual_threshold', 0.90))
            visual_threshold = max(0.0, min(1.0, visual_threshold))

            logo_p = float(request.form.get('logo_p', 0.50))
            logo_p = max(0.0, min(1.0, logo_p))

            w_face = float(request.form.get('w_face', 0.25))
            w_emotion = float(request.form.get('w_emotion', 0.15))
            w_pose = float(request.form.get('w_pose', 0.15))
            w_iqa = float(request.form.get('w_iqa', 0.35))

            # optional: prevent all-zero weights
            if (w_face + w_emotion + w_pose + w_iqa) <= 0:
                w_face, w_emotion, w_pose, w_iqa = 0.25, 0.15, 0.15, 0.35


            thread = threading.Thread(
                target=run_inference,
                args=(
                    model,
                    video_path,
                    fps,
                    redundancy_reduction,
                    visual_threshold,
                    w_face,
                    w_emotion,
                    w_pose,
                    w_iqa,
                    logo_p,
                )
            )
            thread.daemon = True
            thread.start()

            return redirect(url_for('progress'))

    return render_template('index.html')


@app.route('/progress')
def progress():
    """Progress page with real-time updates"""
    return render_template('progress.html')

@app.route('/stream')
def stream():
    def generate():
        while True:
            try:
                status = progress_queue.get(timeout=0.5)
                yield f"data: {json.dumps(status)}\n\n"
                if status.get('stage') in ['Complete', 'Error']:
                    break
            except queue.Empty:
                stage = current_status.get('stage', 'idle')
                if stage not in ('Complete', 'Error', 'idle', ''):
                    yield f"data: {json.dumps(current_status)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/results')
def results():
    keyframes = []
    keyframe_data = []

    csv_path = os.path.join(BASE_DIR, "data", "inference_output", "keyframes", "keyframes.csv")

    try:
        # Load metadata CSV
        if os.path.exists(csv_path):
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("saved_path", "")
                    if filename:
                        # Extract just the filename (not the full path)
                        basename = os.path.basename(filename)
                        
                        keyframes.append(basename)

                        keyframe_data.append({
                            "filename": basename,
                            "final_score": float(row.get("final_score", 0)),
                            "w_iqa": float(row.get("w_iqa", 0)),
                            "w_face": float(row.get("w_face", 0)),
                            "w_emotion": float(row.get("w_emotion", 0)),
                            "w_pose": float(row.get("w_pose", 0)),
                            #"face_signal": float(row.get("face_signal", 0)),
                            #"emotion_signal": float(row.get("emotion_signal", 0)),
                            #"pose_signal": float(row.get("pose_signal", 0)),
                        })

        keyframes = sorted(keyframes)

    except Exception as e:
        print("Error loading metadata:", e)

    return render_template(
        'results.html',
        keyframes=keyframes,
        keyframe_data=keyframe_data
    )


@app.route('/api/status')
def api_status():
    """API endpoint for current status"""
    return jsonify(current_status)


@app.route('/keyframes/<path:filename>')
def serve_keyframe(filename):
    """Serve keyframe images"""
    return send_from_directory(OUTPUT_FOLDER, filename)


# =============================================================================
#  SAM3 Thumbnail Editor Routes
# =============================================================================

@app.route('/api/sam3/health')
def sam3_health():
    """Check if SAM3 service is running."""
    is_healthy = check_sam3_service_health()
    return jsonify({
        "sam3_service": "running" if is_healthy else "offline",
        "healthy": is_healthy
    })


@app.route('/api/segment-player', methods=['POST'])
def segment_player():
    """Segment players using SAM3 microservice."""
    try:
        data = request.get_json()
        keyframe_filename = data.get('keyframe_filename')
        prompt = data.get('prompt', 'soccer player')
        confidence_threshold = data.get('confidence_threshold', 0.5)
        
        if not keyframe_filename:
            return jsonify({"success": False, "error": "keyframe_filename required"}), 400
        
        keyframe_path = os.path.join(OUTPUT_FOLDER, keyframe_filename)
        
        if not os.path.exists(keyframe_path):
            return jsonify({"success": False, "error": f"Keyframe not found: {keyframe_filename}"}), 404
        
        print(f"[API] Segmenting: {keyframe_filename}")
        
        # Call SAM3 microservice
        result = call_sam3_segmentation(
            image_path=keyframe_path,
            prompt=prompt,
            confidence_threshold=confidence_threshold,
            output_dir=MASKS_FOLDER
        )
        
        if not result.get('success'):
            return jsonify({"success": False, "error": result.get('error', 'Segmentation failed')}), 500
        
        # Extract filenames
        mask_filename = os.path.basename(result.get('mask_path', ''))
        preview_filename = os.path.basename(result.get('preview_path', ''))
        
        return jsonify({
            "success": True,
            "mask_url": f"/masks/{mask_filename}" if mask_filename else None,
            "preview_url": f"/masks/{preview_filename}" if preview_filename else None,
            "num_persons": result.get('num_instances', 0),
            "prompt_used": prompt
        })
        
    except ConnectionError:
        return jsonify({"success": False, "error": "SAM3 service is not running"}), 503
    except TimeoutError:
        return jsonify({"success": False, "error": "SAM3 service timeout"}), 504
    except Exception as e:
        print(f"[ERROR] Segmentation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500




@app.route('/api/create-thumbnail', methods=['POST'])
def create_thumbnail_route():
    """
    Create final thumbnail with text overlays.

    Request JSON:
    {
        "keyframe_filename": "...",
        "mask_filename": "...",          // optional
        "text_elements": [...],
        "background_color": "#000000",
        "player_layer": "foreground",    // 'foreground' | 'background'
        "blur_background": false,        // true = blur bg, keep player sharp
        "blur_radius": 12                // blur strength in px
    }
    """
    try:
        data = request.get_json()

        keyframe_filename = data.get('keyframe_filename')
        mask_filename     = data.get('mask_filename')
        #text_elements     = data.get('text_elements', [])
        elements = data.get('elements', [])  # unified list of all graphic elements

        background_color  = data.get('background_color', '#000000')

        player_layer      = data.get('player_layer', 'foreground')
        blur_background   = data.get('blur_background', False)
        blur_radius       = int(data.get('blur_radius', 12))

        if not keyframe_filename:
            return jsonify({"success": False, "error": "keyframe_filename required"}), 400

        keyframe_path = os.path.join(OUTPUT_FOLDER, keyframe_filename)
        if not os.path.exists(keyframe_path):
            return jsonify({"success": False, "error": "Keyframe not found"}), 404

        mask_path = None
        if mask_filename:
            mask_path = os.path.join(MASKS_FOLDER, mask_filename)
            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask not found: {mask_filename}")
                mask_path = None

        # Generate unique output filename
        timestamp       = int(time.time() * 1000)
        base_name       = os.path.splitext(keyframe_filename)[0]
        output_filename = f"{base_name}_thumbnail_{timestamp}.png"
        output_path     = os.path.join(THUMBNAILS_FOLDER, output_filename)

        # Compose thumbnail — pass new params through
        compose_thumbnail(
            image_path       = keyframe_path,
            mask_path        = mask_path,
            elements         = elements, 
            background_color = background_color,
            output_path      = output_path,
            player_layer     = player_layer,
            blur_background  = blur_background,
            blur_radius      = blur_radius,
        )

        return jsonify({
            "success":       True,
            "thumbnail_url": f"/thumbnails/{output_filename}"
        })

    except Exception as e:
        print(f"[ERROR] Thumbnail creation failed: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/masks/<path:filename>')
def serve_mask(filename):
    """Serve segmentation mask images."""
    return send_from_directory(MASKS_FOLDER, filename)


@app.route('/thumbnails/<path:filename>')
def serve_thumbnail_file(filename):
    """Serve generated thumbnail images."""
    return send_from_directory(THUMBNAILS_FOLDER, filename)



@app.route('/api/fotbollplay/events')
def get_fotbollplay_events():
    """
    Proxy endpoint to fetch events from FotbollPlay API with caching.
    """
    # Check cache first
    cache_key = 'fotbollplay_events'
    if cache_key in METADATA_CACHE:
        cached_data, cached_time = METADATA_CACHE[cache_key]
        if datetime.now() - cached_time < timedelta(seconds=CACHE_DURATION):
            return jsonify(cached_data)
    
    try:
        # Fetch from FotbollPlay API
        response = requests.get(
            f"{API}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        # Cache the response
        METADATA_CACHE[cache_key] = (data, datetime.now())
        
        return jsonify(data)
        
    except requests.RequestException as e:
        return jsonify({
            "success": False,
            "error": f"Failed to fetch from FotbollPlay API: {str(e)}"
        }), 500



@app.route('/api/video-metadata/<keyframe_filename>')
def get_video_metadata(keyframe_filename):
    """
    Get metadata for a specific keyframe by matching video_asset_id to a
    FotbollPlay event via a pre-built index.

    video_id extraction (priority order):
      1. Filename pattern:  video_17534_rank01_… → "17534"
      2. keyframes.csv saved_path column (handles mixed Windows/POSIX paths)

    Matching uses the index built by build_video_index():
        video_asset_id → event (O(1), full dataset, cached)
    """
    try:
        # ── 1. Extract video_id ───────────────────────────────────────────────
        video_id = None
        bare = keyframe_filename.replace('\\', '/').split('/')[-1]

        if 'video_' in bare:
            try:
                video_id = bare.split('video_')[1].split('_')[0]
            except IndexError:
                pass

        if not video_id:
            video_id = _lookup_video_id_from_csv(bare)

        print(f"[metadata] keyframe={bare}  resolved video_id={video_id!r}")

        if not video_id:
            return jsonify({"success": False, "error": "Could not extract video_id"}), 400

        # ── 2. O(1) index lookup ──────────────────────────────────────────────
        index = get_video_index()
        matching_events = index.get(str(video_id), [])

        if not matching_events:
            print(f"[INFO] video_id={video_id!r} not found in index.")
            return jsonify({
                "success": True,
                "video_id": video_id,
                "matched": False,
                "metadata": None
            })

        # Take the first match (index was built in chronological API order)
        event = matching_events[0]

        # ── 3. Build rich metadata (same fields as before) ────────────────────
        game          = event.get('playlist', {}).get('game', {})
        home_team     = game.get('home_team', {})
        visiting_team = game.get('visiting_team', {})
        tag           = event.get('tag', {})

        metadata = {
            "score":      event.get('score', '0-0'),
            "game_time":  format_game_time(event.get('game_time', 0)),
            "game_phase": event.get('game_phase', ''),
            "event_type": tag.get('action', 'highlight'),
            "scorer":     tag.get('player_name') or tag.get('scorer') or '',

            "home_team":           home_team.get('name', ''),
            "home_team_short":     home_team.get('short_name', ''),
            "home_team_logo":      home_team.get('logo_url', ''),

            "visiting_team":       visiting_team.get('name', ''),
            "visiting_team_short": visiting_team.get('short_name', ''),
            "visiting_team_logo":  visiting_team.get('logo_url', ''),

            "stadium":    game.get('stadium_name', ''),
            "tournament": game.get('tournament_name', ''),
            "date":       game.get('date', ''),
            "attendance": game.get('attendance'),

            "video_url":     event.get('playlist', {}).get('video_url', ''),
            "thumbnail_url": event.get('playlist', {}).get('thumbnail_url', ''),
            "description":   event.get('playlist', {}).get('description', ''),
        }

        return jsonify({
            "success": True,
            "video_id": video_id,
            "matched": True,
            "metadata": metadata
        })

    except Exception as e:
        print(f"[ERROR] get_video_metadata: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def _lookup_video_id_from_csv(bare_filename: str):
    """
    Read video_id from keyframes.csv by matching the bare filename against
    the saved_path column.  Handles mixed Windows/POSIX slashes.
    """
    csv_path = os.path.join(BASE_DIR, "data", "inference_output", "keyframes", "keyframes.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                saved = row.get('saved_path', '').replace('\\', '/').split('/')[-1]
                if saved == bare_filename:
                    return row.get('video_id') or None
    except Exception as e:
        print(f"[WARN] CSV lookup failed: {e}")
    return None


def _fetch_goal_events_for_range(from_date: str, to_date: str, label: str) -> list:
    """
    Fetch all goal events for a single date range (paginated).
    Runs in its own thread so two seasons can be fetched in parallel.
    """
    events = []
    offset = 0
    page_size = 100  # max out page size to minimise round-trips

    while True:
        url = (
            "https://api.fotbollplay.se/allsvenskan/event?"
            f"from_date={from_date}&"
            f"to_date={to_date}&"
            "min_rating=1&"
            "tags=%7B%22action%22%3A%22goal%22%7D&"
            f"count={page_size}&from={offset}"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"[Index:{label}] Fetch error at offset={offset}: {exc}")
            break

        page = data.get("events", []) if isinstance(data, dict) else data
        if not page:
            break

        events.extend(page)
        print(f"[Index:{label}] {len(events)} events so far…")

        if len(page) < page_size:
            break

        offset += page_size

    print(f"[Index:{label}] Done — {len(events)} events.")
    return events


def build_video_index():
    """
    Fetch goal events for 2024 and 2025 seasons in parallel, then build:
        video_asset_id (str) → list[event]

    Using tags=goal keeps the dataset small (~700 events/season vs 7000+).
    Two threads run simultaneously so total time ≈ one season's fetch time.
    """
    global VIDEO_INDEX, INDEX_LAST_UPDATED

    print("[Index] Building video index (parallel fetch, goals only)…")

    results: dict[str, list] = {"2024": [], "2025": []}

    def fetch_2024():
        results["2024"] = _fetch_goal_events_for_range(
            "2024-01-01T00:00:00.000Z", "2025-01-01T00:00:00.000Z", "2024"
        )

    def fetch_2025():
        results["2025"] = _fetch_goal_events_for_range(
            "2025-01-01T00:00:00.000Z", "2026-01-01T00:00:00.000Z", "2025"
        )

    t1 = threading.Thread(target=fetch_2024)
    t2 = threading.Thread(target=fetch_2025)
    t1.start(); t2.start()
    t1.join();  t2.join()

    all_events = results["2024"] + results["2025"]

    # Build index: video_asset_id → list of parent events
    index: dict = {}
    for event in all_events:
        for pe in event.get("playlist", {}).get("events", []):
            vid = pe.get("video_asset_id")
            if vid:
                index.setdefault(str(vid), []).append(event)

    VIDEO_INDEX = index
    INDEX_LAST_UPDATED = time.time()
    print(
        f"[Index] Done. Indexed {len(VIDEO_INDEX)} unique video_asset_ids "
        f"across {len(all_events)} total events "
        f"(2024: {len(results['2024'])}, 2025: {len(results['2025'])})."
    )


def get_video_index() -> dict:
    """Return the cached index, rebuilding it if stale or empty."""
    global VIDEO_INDEX, INDEX_LAST_UPDATED

    if (
        not VIDEO_INDEX
        or INDEX_LAST_UPDATED is None
        or time.time() - INDEX_LAST_UPDATED > INDEX_TTL
    ):
        build_video_index()

    return VIDEO_INDEX


def format_game_time(seconds):
    """Convert seconds to MM:SS format"""
    if not seconds:
        return "00:00"
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"

if __name__ == '__main__':
    app.run(debug=False, threaded=True)