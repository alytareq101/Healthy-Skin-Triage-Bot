from dotenv import load_dotenv
load_dotenv()  # ✅ MUST be first

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from model import predict_risk
from rag import retrieve_context

# -------------------------
# OpenAI Client
# -------------------------
client = OpenAI()  # ✅ reads OPENAI_API_KEY automatically

# -------------------------
# App Setup
# -------------------------
app = FastAPI(title="Healthy Skin Triage Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# API Endpoint
# -------------------------
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # 1️⃣ Read image
        image_bytes = await file.read()

        # 2️⃣ CNN Prediction
        cnn_result = predict_risk(image_bytes)
        risk = cnn_result["risk"]
        confidence = cnn_result["confidence"]

        # 3️⃣ Retrieve medical context (RAG only for High Risk)
        context = ""
        if risk == "High Risk":
            context = retrieve_context(
                "Dermatology guidelines for high-risk skin lesion"
            )

        # 4️⃣ Build OpenAI prompt
        prompt = f"""You are a dermatology assistant.

CNN assessment:
- Risk level: {risk}
- Confidence: {confidence}
"""

        if context:
            prompt += f"\nRelevant medical guidelines:\n{context}\n"

        prompt += "Explain the result in simple, patient-friendly language.\nInclude a medical disclaimer."

        # 5️⃣ OpenAI Explanation
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt
        )

        explanation = response.output_text

        return {
            "risk": risk,
            "confidence": confidence,
            "explanation": explanation
        }

    except Exception as e:
        print("❌ analyze-image error:", e)
        return {
            "error": "Internal Server Error",
            "details": str(e)
        }
