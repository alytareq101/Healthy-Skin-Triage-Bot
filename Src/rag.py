
# rag.py
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------
# Environment
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment")

# -------------------------
# Vector DB Config
# -------------------------
VECTOR_DB_PATH = "vectorstore/dermatology_faiss"

INDEX_FILE = os.path.join(VECTOR_DB_PATH, "index.faiss")
PKL_FILE = os.path.join(VECTOR_DB_PATH, "index.pkl")

# -------------------------
# Embeddings (must match build time)
# -------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# -------------------------
# Load Vector Database (ONCE)
# -------------------------
_vector_db = None

def load_vector_db():
    global _vector_db

    if _vector_db is not None:
        return _vector_db

    if not os.path.exists(INDEX_FILE) or not os.path.exists(PKL_FILE):
        raise RuntimeError(
            "❌ Vector database not found. Make sure index.faiss and index.pkl exist."
        )

    print("📦 Loading FAISS vector database...")
    _vector_db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ Vector database loaded")
    return _vector_db

# -------------------------
# Retrieve Relevant Context
# -------------------------
def retrieve_context(query: str, k: int = 4) -> str:
    """
    Retrieve top-k relevant dermatology guideline chunks
    """
    db = load_vector_db()

    docs = db.similarity_search(query, k=k)

    if not docs:
        return "No relevant dermatology guidelines found."

    context = "\n\n".join(
        f"- {doc.page_content.strip()}"
        for doc in docs
    )

    return context

