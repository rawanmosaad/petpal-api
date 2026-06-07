# PetPal FastAPI AI Service

Runs sentiment analysis and dog photo matching on port `8000`.

## Requirements

Use Python `3.11` or `3.12`. The dog photo matcher depends on TensorFlow, and the pinned TensorFlow package does not install on Python `3.13`.

```powershell
cd petpal-api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

If `py -3.12` is not available, install Python 3.12 first, then rerun the commands above.

## Model Files

Dog matching needs these files:

- `models/dog_face_embedding_model.keras`
- `models/dog_gallery_index.npz`

The service health at `http://127.0.0.1:8000/` reports whether the dog model loaded and includes the exact load error if it did not.
