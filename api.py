import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from engine import RAGEngine
from persona_manager import PersonaManager

load_dotenv()

app = FastAPI(title="Tawuniya RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHUNKS_FOLDER = os.path.join("splitted_chunks", "splitted_chunks")
engine = RAGEngine(
    chunks_folder=CHUNKS_FOLDER,
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
)

# ── PersonaManager lives here ──────────────────────────────────────────────
persona_manager = PersonaManager()


# ── Monkey-patch engine to use PersonaManager instead of hardcoded REPLY_MODES
# This means every call to engine.classify_intent() and engine._build_mode_block()
# will now read from personas.json — no engine.py changes needed.

def _pm_classify_intent(self, query: str) -> str:
    valid_ids = persona_manager.valid_ids()
    fallback  = valid_ids[0] if valid_ids else "knowledge"
    try:
        prompt = persona_manager.build_classification_prompt()
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user",   "content": query},
            ],
            temperature=0,
            max_tokens=30,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content.strip())
        intent = result.get("intent", fallback)
        return intent if intent in valid_ids else fallback
    except Exception as e:
        print(f"[WARN] Intent classification failed: {e}")
        return fallback


def _pm_build_mode_block(self, intent: str, lang: str) -> str:
    return persona_manager.build_mode_block(intent)


# Bind the patched methods onto the live engine instance
import types
engine.classify_intent    = types.MethodType(_pm_classify_intent,    engine)
engine._build_mode_block  = types.MethodType(_pm_build_mode_block,   engine)


# ── Request / Response models ──────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    debug: bool = False


class PersonaExample(BaseModel):
    user: str
    assistant: str


class PersonaCreate(BaseModel):
    name: str
    label: str
    trigger_description: str
    instruction: str
    examples: List[PersonaExample] = []


class PersonaUpdate(BaseModel):
    name: str
    label: str
    trigger_description: str
    instruction: str
    examples: List[PersonaExample] = []


# ── Chat endpoints ─────────────────────────────────────────────────────────

@app.post("/ask")
def ask(req: AskRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    answer, context = engine.answer_question(
        req.query, debug=req.debug, return_context=True
    )
    lang = engine.detect_language(req.query)
    followups = engine.generate_followup_questions(req.query, answer, context, lang)

    return {"answer": answer, "followup_questions": followups, "lang": lang}


@app.post("/clear")
def clear_history():
    engine.clear_history()
    return {"message": "Conversation history cleared."}


@app.get("/stats")
def stats():
    return engine.get_statistics()


# ── Persona CRUD endpoints ─────────────────────────────────────────────────

@app.get("/personas")
def list_personas():
    return persona_manager.get_all()


@app.post("/personas", status_code=201)
def create_persona(body: PersonaCreate):
    data = {
        "name": body.name,
        "label": body.label,
        "trigger_description": body.trigger_description,
        "instruction": body.instruction,
        "examples": [e.dict() for e in body.examples],
    }
    return persona_manager.create(data)


@app.get("/personas/{persona_id}")
def get_persona(persona_id: str):
    persona = persona_manager.get_by_id(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found.")
    return persona


@app.put("/personas/{persona_id}")
def update_persona(persona_id: str, body: PersonaUpdate):
    data = {
        "name": body.name,
        "label": body.label,
        "trigger_description": body.trigger_description,
        "instruction": body.instruction,
        "examples": [e.dict() for e in body.examples],
    }
    updated = persona_manager.update(persona_id, data)
    if not updated:
        raise HTTPException(status_code=404, detail="Persona not found.")
    return updated


@app.delete("/personas/{persona_id}", status_code=204)
def delete_persona(persona_id: str):
    if not persona_manager.delete(persona_id):
        raise HTTPException(status_code=404, detail="Persona not found.")


# ── Serve frontend ─────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/admin")
def admin():
    return FileResponse("static/admin.html")
