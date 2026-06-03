from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import numpy as np
import io

app = FastAPI(title="PetPal API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(__file__)
SENTIMENT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "petpal_roberta_model")
DOG_MODEL_PATH       = os.path.join(BASE, "models", "dog_face_embedding_model.keras")
GALLERY_PATH         = os.path.join(BASE, "models", "dog_gallery_index.npz")

sentiment_pipeline = None
dog_model          = None
gallery_embeddings = None
gallery_dog_ids    = None
SENTIMENT_READY    = False
DOG_READY          = False

@app.on_event("startup")
def load_models():
    global sentiment_pipeline, SENTIMENT_READY
    global dog_model, gallery_embeddings, gallery_dog_ids, DOG_READY

    if os.path.isdir(SENTIMENT_MODEL_PATH):
        try:
            from transformers import pipeline as hf_pipeline
            sentiment_pipeline = hf_pipeline(
    "text-classification",
    model=SENTIMENT_MODEL_PATH,
    tokenizer=SENTIMENT_MODEL_PATH,
    device=-1,
    top_k=None,
    trust_remote_code=True,
)
            SENTIMENT_READY = True
            print("✅ Sentiment model loaded.")
        except Exception as e:
            print(f"⚠️  Sentiment model failed: {e}")
    else:
        print("⏳ Sentiment model not found — add files to models/petpal_roberta_model/")

    if os.path.exists(DOG_MODEL_PATH) and os.path.exists(GALLERY_PATH):
        try:
            import tensorflow as tf
            dog_model = tf.keras.models.load_model(DOG_MODEL_PATH)
            data = np.load(GALLERY_PATH, allow_pickle=True)
            gallery_embeddings = data["embeddings"]
            gallery_dog_ids    = data["dog_ids"]
            DOG_READY = True
            print("✅ Dog face model loaded.")
        except Exception as e:
            print(f"⚠️  Dog model failed: {e}")
    else:
        print("⏳ Dog model not found — add files to models/")


@app.get("/")
def root():
    return {
        "status": "running",
        "sentiment_ready": SENTIMENT_READY,
        "dog_model_ready": DOG_READY,
    }


# ── Sentiment ─────────────────────────────────────────────────────────────────
class SentimentRequest(BaseModel):
    text: str

@app.post("/api/sentiment")
def analyse_sentiment(body: SentimentRequest):
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="text cannot be empty")

    if not SENTIMENT_READY:
        return {
            "ready": False,
            "label": None,
            "score": None,
            "scores": {},
            "message": "Sentiment model not loaded. Add files to models/petpal_roberta_model/"
        }

    try:
        results = sentiment_pipeline(body.text.strip())[0]
        best    = max(results, key=lambda x: x["score"])
        scores  = {r["label"].lower(): round(r["score"], 4) for r in results}
        return {
            "ready": True,
            "label": best["label"].lower(),
            "score": round(best["score"], 4),
            "scores": scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dog Match ─────────────────────────────────────────────────────────────────
IMG_SIZE = 224

@app.post("/api/dog-match")
async def dog_match(file: UploadFile = File(...), threshold: float = 0.75):
    if not DOG_READY:
        return {
            "ready": False,
            "matched": False,
            "results": [],
            "message": "Dog model not loaded. Add files to models/"
        }

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File must be an image")

    try:
        import tensorflow as tf
        from PIL import Image

        contents   = await file.read()
        pil_img    = Image.open(io.BytesIO(contents)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array  = np.array(pil_img, dtype=np.float32)
        img_tensor = tf.expand_dims(img_array, axis=0)

        query_embedding = dog_model.predict(img_tensor, verbose=0)[0]
        similarities    = np.dot(gallery_embeddings, query_embedding)

        result_map = {}
        for i, sim in enumerate(similarities):
            dog_id = str(gallery_dog_ids[i])
            if dog_id not in result_map or sim > result_map[dog_id]:
                result_map[dog_id] = float(sim)

        sorted_results = sorted(result_map.items(), key=lambda x: x[1], reverse=True)[:5]
        top_results    = [{"dog_id": d, "similarity": round(s, 4)} for d, s in sorted_results]
        best_sim       = top_results[0]["similarity"] if top_results else 0
        matched        = best_sim >= threshold

        return {
            "ready":      True,
            "matched":    matched,
            "threshold":  threshold,
            "best_match": top_results[0] if matched else None,
            "results":    top_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dog Register ──────────────────────────────────────────────────────────────
@app.post("/api/dog-register")
async def dog_register(file: UploadFile = File(...), dog_id: str = ""):
    if not DOG_READY:
        return {"ready": False, "message": "Dog model not loaded."}

    if not dog_id.strip():
        raise HTTPException(status_code=422, detail="dog_id is required")

    try:
        import tensorflow as tf
        from PIL import Image
        global gallery_embeddings, gallery_dog_ids

        contents      = await file.read()
        pil_img       = Image.open(io.BytesIO(contents)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array     = np.array(pil_img, dtype=np.float32)
        img_tensor    = tf.expand_dims(img_array, axis=0)
        new_embedding = dog_model.predict(img_tensor, verbose=0)[0]

        gallery_embeddings = np.vstack([gallery_embeddings, new_embedding])
        gallery_dog_ids    = np.append(gallery_dog_ids, dog_id.strip())

        np.savez(
            GALLERY_PATH,
            embeddings=gallery_embeddings,
            dog_ids=gallery_dog_ids,
            image_paths=np.array(["uploaded"] * len(gallery_dog_ids)),
        )

        return {"ready": True, "registered": True, "dog_id": dog_id.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))