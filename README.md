# Streamlit Deployment for `my_model.h5`

This repository wraps the pre-trained `my_model.h5` Keras CNN in a ready-to-deploy Streamlit experience. The model expects **32×32 RGB images** (CIFAR-10 style) and outputs probabilities for ten classes. The app handles preprocessing, runs inference, and visualizes the confidence per class.

## Features

- 📁 Upload JPG/PNG images and preview them instantly.
- 🔄 Automatic EXIF-aware rotation fix, RGB conversion, and 32×32 resizing.
- 🧠 Cached TensorFlow model loading for fast repeat predictions.
- 📊 Top-k prediction table plus full probability bar chart.
- 🏷️ Sidebar control to override class labels (JSON array or comma-separated list).

## Requirements

- Python **3.10** or **3.13** (tested locally on 3.10 and in Streamlit Cloud's default 3.13 builder).
- Packages listed in `requirements.txt` (TensorFlow 2.20.0 + Streamlit + supporting libs).

> Streamlit Cloud's "uv" builder currently defaults to Python 3.13. The pinned dependency set is compatible with that environment. If you prefer Python 3.10, keep `runtime.txt` in version control or add a `.python-version` file with `3.10.12`.

## Local setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Then open the provided URL (default: <http://localhost:8501>) in your browser.

Place `my_model.h5` alongside `app.py` (already included by default). Upload an image to see predictions. Adjust the sidebar label field to map the ten output indices to your preferred class names.

## Deploy to Streamlit Cloud

1. Push this repository (including `my_model.h5`) to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io/), choose **New app** and connect the repo.
3. Select the branch and set the main file path to `app.py`.
4. The cloud build installs everything in `requirements.txt`. With the default uv builder you don't need to set a Python version, but you can pin one via the app's advanced settings or by committing `.python-version`.
5. Once live, upload images via the web UI to generate predictions.

## Customization

- **Class labels:** Replace the default CIFAR-10 labels by editing `DEFAULT_CLASS_NAMES` in `app.py`, or use the sidebar JSON/comma input at runtime.
- **Image size:** If your model expects a different resolution, update `IMAGE_SIZE` and the resize logic in `preprocess`.
- **Model path:** Set `MODEL_PATH` if you relocate or rename the `.h5` file. Remote storage (S3, GCS, etc.) can be supported by downloading the file at startup.

## Troubleshooting

- **ImportError: No module named `tensorflow`:** Ensure the virtual environment activated for the session matches the dependencies (Python ≥3.10) and reinstall via `pip install -r requirements.txt`.
- **Model not found:** Ensure `my_model.h5` is committed and available at runtime. Streamlit Cloud requires the file to be under 100 MB.
- **Different number of classes:** Update `DEFAULT_CLASS_NAMES` (length must match your model's output dimension) and retrain if necessary.

Happy deploying! 🎈
