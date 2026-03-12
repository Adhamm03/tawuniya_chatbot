import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from engine import RAGEngine

app = FastAPI(title="Tawuniya RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine once at startup
CHUNKS_FOLDER = os.path.join("splitted_chunks", "splitted_chunks")
engine = RAGEngine(
    chunks_folder=CHUNKS_FOLDER,
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
)


class AskRequest(BaseModel):
    query: str
    debug: bool = False


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    answer, context = engine.answer_question(
        req.query, debug=req.debug, return_context=True
    )

    lang = engine.detect_language(req.query)
    followups = engine.generate_followup_questions(req.query, answer, context, lang)

    return {
        "answer": answer,
        "followup_questions": followups,
        "lang": lang,
    }


@app.post("/clear")
def clear_history():
    engine.clear_history()
    return {"message": "Conversation history cleared."}


@app.get("/stats")
def stats():
    return engine.get_statistics()


# ── Serve frontend ──────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")
