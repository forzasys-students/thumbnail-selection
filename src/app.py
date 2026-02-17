from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
import os
import subprocess
import threading
import queue
import json
import csv
from werkzeug.utils import secure_filename
import time

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


def run_inference(model, video_path):
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
        
        # Start the subprocess with UTF-8 encoding
        process = subprocess.Popen(
            [bash_executable, SCRIPT_PATH, "--model", model, "--video", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of crashing
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
    if request.method == 'POST':
        model = request.form['model']
        video_file = request.files['video']

        if video_file:
            filename = secure_filename(video_file.filename)

            upload_dir = os.path.join(BASE_DIR, "data")
            os.makedirs(upload_dir, exist_ok=True) 

            video_path = os.path.join(upload_dir, filename)
            video_file.save(video_path)

            thread = threading.Thread(target=run_inference, args=(model, video_path))
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
    """Server-Sent Events stream for progress updates"""
    def generate():
        last_status = {}
        while True:
            try:
                status = progress_queue.get(timeout=0.5)
                yield f"data: {json.dumps(status)}\n\n"
                last_status = status
                
                if status.get('stage') in ['Complete', 'Error']:
                    break
            except queue.Empty:
                # Send keepalive with current status
                if current_status:
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
                            "face_signal": float(row.get("face_signal", 0)),
                            "emotion_intensity": float(row.get("emotion_intensity", 0)),
                            "pose_signal": float(row.get("pose_signal", 0)),
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
        text_elements     = data.get('text_elements', [])
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
            text_elements    = text_elements,
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


if __name__ == '__main__':
    app.run(debug=False, threaded=True)