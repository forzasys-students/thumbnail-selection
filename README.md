# Thumbnail Selection Project Setup

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
