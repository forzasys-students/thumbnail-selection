from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
import os
import subprocess
import threading
import queue
import json
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to inference script
SCRIPT_PATH = os.path.join(BASE_DIR, "run_inference_pipeline.sh")

# Output folder for keyframes
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "inference_output", "keyframes")

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
    print(f"[DEBUG] Looking for keyframes in: {OUTPUT_FOLDER}")
    print(f"[DEBUG] OUTPUT_FOLDER exists: {os.path.exists(OUTPUT_FOLDER)}")
    
    try:
        # Walk through all subdirectories to find images
        for root, dirs, files in os.walk(OUTPUT_FOLDER):
            print(f"[DEBUG] Checking directory: {root}")
            print(f"[DEBUG] Files found: {files}")
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Get relative path from OUTPUT_FOLDER
                    rel_path = os.path.relpath(os.path.join(root, file), OUTPUT_FOLDER)
                    keyframes.append(rel_path.replace('\\', '/'))  # Convert to forward slashes for URLs
                    print(f"[DEBUG] Added keyframe: {rel_path}")
        keyframes = sorted(keyframes)
        print(f"[DEBUG] Total keyframes found: {len(keyframes)}")
    except FileNotFoundError as e:
        print(f"[DEBUG] FileNotFoundError: {e}")
        keyframes = []
    except Exception as e:
        print(f"[DEBUG] Unexpected error: {e}")
        keyframes = []

    return render_template('results.html', keyframes=keyframes)


@app.route('/api/status')
def api_status():
    """API endpoint for current status"""
    return jsonify(current_status)


@app.route('/keyframes/<path:filename>')
def serve_keyframe(filename):
    """Serve keyframe images"""
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=False, threaded=True)