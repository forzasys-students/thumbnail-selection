# AI-based Thumbnail Generation for Soccer Clips using Multi-Signal Analysis


## Abstract

Thumbnail selection is an important part of sports video presentation. It creates the first impression of a video, and often decides whether a viewer clicks on the video or scrolls past it. Manual thumbnail selection can give good results, but it is slow and subjective, also difficult to scale when dealing with many clips that needs to be published quickly. Existing automatic methods still struggle with this because thumbnail quality depends on more than just choosing a clear or central frame. A good soccer thumbnail also needs to reflect the event, show the right camera perspective, and capture details such as emotion, player focus, and image quality. This thesis presents an AI-based pipeline for automatic thumbnail selection in soccer goal clips. The system extracts frames from a clip, classifies the camera shot-type, prioritizes close-up segments, removes low-quality and redundant frames, filters out frames containing broadcast logos and overlays, and ranks the remaining frames using several visual signals. These signals include face detection, facial emotion, pose estimation, and image quality assessment. The system is evaluated through various experiments on the individual components, cross-league testing of the camera shot-type classifier, runtime and resource usage analysis, and a user study comparing different thumbnail selection strategies. The results show that the pipeline can produce useful thumbnail candidates and reduce the amount of manual searching needed from an editor. The strongest results are achieved when AI-selected candidates are combined with human refinement and graphical overlays. This suggests that the most practical solution is a human-in-the-loop workflow, where AI handles fast candidate generation and editors keep control over the final thumbnail.

## Demo Video

Click the image below to watch a video that demonstrates the graphical user interface and shows how the system processes a soccer goal clip, ranks thumbnail candidates, and allows refinement of the final thumbnail

[![AI-Based Thumbnail Generation Demo](https://img.youtube.com/vi/6L3BeG1f6Eo/maxresdefault.jpg)](https://www.youtube.com/watch?v=6L3BeG1f6Eo)


# Project Setup

## 1. Clone the Project

Open **Anaconda Prompt** and run:

```bash
git clone https://github.com/forzasys-students/thumbnail-selection.git
cd thumbnail-selection
```

---

## 2. Create the Conda Environments

Run these commands from the project root, where the `.yml` files are located:

```bash
conda env create -f thumbnailselection_env.yml
conda env create -f sam3_env.yml
```

---

## 3. Install CUDA PyTorch for `thumbnailselection_env`

Activate the environment:

```bash
conda activate thumbnailselection_env
```

Install PyTorch and Torchvision with CUDA 12.1:

```bash
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Test that GPU works:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## 4. Install CUDA PyTorch for `sam3_env`

Open a second **Anaconda Prompt**.

Go to the project folder:

```bash
cd thumbnail-selection
```

Activate the SAM3 environment:

```bash
conda activate sam3_env
```

Install PyTorch and Torchvision with CUDA 12.6:

```bash
python -m pip install torch==2.10.0+cu126 torchvision==0.25.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

Test that GPU works:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

# Running the Project

You need two **Anaconda Prompt** terminals open.

---

## Terminal 1: Run the Flask App

```bash
conda activate thumbnailselection_env
cd thumbnail-selection
cd src
python -m flask run
```

The Flask app should run on:

```text
http://127.0.0.1:5000
```

If `flask run` works normally, this can also be used:

```bash
flask run
```

---

## Terminal 2: Run the SAM3 Service

```bash
conda activate sam3_env
cd thumbnail-selection
cd src
cd sam3
python sam3_service.py
```

The SAM3 service should run on:

```text
http://127.0.0.1:8001
```

---

# Notes

- The `thumbnailselection_env` is used for the main Flask pipeline.
- The `sam3_env` is used separately for the SAM3 segmentation microservice.
- Both terminals must stay open while using the full application.
- If CUDA is working, `torch.cuda.is_available()` should print `True`.
