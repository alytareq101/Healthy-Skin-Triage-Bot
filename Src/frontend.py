# frontend.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("🩺 Healthy Skin Triage Bot")

uploaded_file = st.file_uploader(
    "Upload skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.write("Uploading image...")

    # ✅ CORRECT way to send file to FastAPI
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file,
            uploaded_file.type
        )
    }

    response = requests.post(
        f"{API_URL}/analyze-image",
        files=files
    )

    # 🔍 DEBUG output (VERY IMPORTANT)
    st.write("Status code:", response.status_code)
    st.write("Raw response:", response.text)

    # ✅ SAFE JSON parsing
    if response.headers.get("content-type", "").startswith("application/json"):
        result = response.json()

        if "error" in result:
            st.error(result["error"])
        else:
            st.success("Prediction successful!")
            st.write(result)
    else:
        st.error("Backend did not return JSON.")
