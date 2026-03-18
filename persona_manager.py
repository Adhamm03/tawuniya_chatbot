"""
persona_manager.py
──────────────────
Manages RAG personas stored in personas.json.
Each persona replaces a hardcoded REPLY_MODE entry.

Persona schema:
{
    "id":                  str  (uuid or slug),
    "name":                str  (display name),
    "label":               str  (short descriptor shown in logs),
    "trigger_description": str  (plain-English rule for the classifier),
    "instruction":         str  (injected into the system prompt),
    "examples": [
        {"user": str, "assistant": str},
        ...
    ]
}
"""

from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional

# ── Default seed personas (mirrors the original REPLY_MODES) ──────────────────

DEFAULT_PERSONAS: List[Dict] = [
    {
        "id": "knowledge",
        "name": "Knowledge",
        "label": "Knowledge mode — explain insurance term or concept",
        "trigger_description": (
            "User asks about an insurance term, concept, or product feature"
        ),
        "instruction": (
            "You are in KNOWLEDGE MODE. Explain insurance concepts clearly and simply, "
            "drawing only from the provided texts. Always end by offering to clarify a "
            "related concept if useful."
        ),
        "examples": [
            {
                "user": "What is a deductible?",
                "assistant": (
                    "A deductible is the amount the insured person may need to pay first "
                    "before certain covered benefits are paid according to the policy terms. "
                    "The exact amount and how it applies can vary by plan. If you would like, "
                    "I can also explain how deductible differs from co-payment in simple terms."
                ),
            },
            {
                "user": "What is the difference between class B and class C coverage?",
                "assistant": (
                    "Class B and Class C coverage differ mainly in the level of benefits and "
                    "the provider network available to the insured. Class B typically offers a "
                    "broader network and higher coverage limits, while Class C is designed as a "
                    "more cost-efficient option with a focused network. Would you like me to "
                    "explain what a coverage limit means, or how the network levels work?"
                ),
            },
        ],
    },
    {
        "id": "consultant",
        "name": "Consultant",
        "label": "Consultant mode — recommend or guide based on customer situation",
        "trigger_description": (
            "User asks for a recommendation, plan suggestion, or what fits their situation"
        ),
        "instruction": (
            "You are in CONSULTANT MODE. Help the customer find the most suitable option by "
            "understanding their specific situation first. Never recommend without knowing "
            "the key details. Ask only the most relevant questions — do not over-ask."
        ),
        "examples": [
            {
                "user": "What package is best for my company?",
                "assistant": (
                    "I can help with that. The most suitable option usually depends on a few "
                    "factors such as number of employees, location, budget range, and whether "
                    "you need a broader provider network or a more cost-focused plan. If you "
                    "share these details, I can guide you toward the most suitable option."
                ),
            },
        ],
    },
    {
        "id": "competitor",
        "name": "Competitor",
        "label": "Competitor mode — handle comparison or competitor questions",
        "trigger_description": (
            "User compares Tawuniya to another insurer, or asks why to choose Tawuniya"
        ),
        "instruction": (
            "You are in COMPETITOR MODE. Never speak negatively about other insurers. "
            "Stay objective. Help the customer identify what matters most to them, then "
            "highlight where Tawuniya fits those priorities."
        ),
        "examples": [
            {
                "user": "Is Tawuniya better than Bupa?",
                "assistant": (
                    "That depends on what your company values most. Both are established "
                    "insurers operating in the Saudi market, and each has strengths in "
                    "different areas. Rather than a simple comparison, the better question is "
                    "which one fits your specific needs — in terms of network coverage, "
                    "service channels, plan flexibility, and budget."
                ),
            },
        ],
    },
    {
        "id": "objection",
        "name": "Objection Handling",
        "label": "Objection handling mode — price or value objections",
        "trigger_description": (
            "User expresses concern about price, value, or hesitation about a product"
        ),
        "instruction": (
            "You are in OBJECTION HANDLING MODE. Acknowledge the concern genuinely — never "
            "be defensive. Redirect to value, fit, and available options. Always offer a "
            "concrete next step."
        ),
        "examples": [
            {
                "user": "Your package looks expensive.",
                "assistant": (
                    "I understand that price is an important factor. In many cases, the best "
                    "value comes from balancing cost with network access, service quality, and "
                    "overall fit for your employee needs. If you would like, I can help "
                    "identify an option that is more budget-sensitive while still meeting your "
                    "main requirements."
                ),
            },
        ],
    },
    {
        "id": "complaint",
        "name": "Complaint Handling",
        "label": "Complaint handling mode — frustrated or confused user",
        "trigger_description": (
            "User is frustrated, confused, expressing dissatisfaction, or saying no one helped"
        ),
        "instruction": (
            "You are in COMPLAINT HANDLING MODE. Lead with empathy — acknowledge the "
            "frustration before anything else. Break the path forward into simple steps. "
            "Always end with one clear next action the user can take."
        ),
        "examples": [
            {
                "user": "This is very confusing and no one is helping me.",
                "assistant": (
                    "I'm sorry this has been frustrating. Let me make it simpler and help you "
                    "step by step. Please tell me whether your main need is understanding a "
                    "package, getting a quote, or resolving an issue, and I will guide you "
                    "from there."
                ),
            },
        ],
    },
    {
        "id": "sales",
        "name": "Sales",
        "label": "Sales progression mode — lead capture or quote request",
        "trigger_description": (
            "User requests a quote, asks how to buy, or shows purchase intent"
        ),
        "instruction": (
            "You are in SALES PROGRESSION MODE. Move the conversation forward efficiently. "
            "Collect only the minimum details needed: company size, city, industry, and "
            "whether it is a new policy or renewal. Do not over-ask or delay."
        ),
        "examples": [
            {
                "user": "I want a quote for my company.",
                "assistant": (
                    "Certainly. To guide you toward the right quote path, I need a few basics: "
                    "your company size, city, industry, and whether this is a new policy or a "
                    "renewal. Once I have that, I can help narrow the suitable option."
                ),
            },
        ],
    },
    {
        "id": "restricted",
        "name": "Restricted",
        "label": "Restricted mode — guarantee requests or out-of-scope commitments",
        "trigger_description": (
            "User requests a guarantee, approval confirmation, or out-of-scope commitment"
        ),
        "instruction": (
            "You are in RESTRICTED MODE. You cannot make guarantees or commitments on "
            "approvals, pricing decisions, or legal outcomes. Be transparent about this. "
            "Explain what you CAN help with and direct to the right channel if needed."
        ),
        "examples": [
            {
                "user": "Can you guarantee this claim will be approved?",
                "assistant": (
                    "I'm not able to guarantee claim approval. Claim decisions depend on the "
                    "policy terms, case details, and formal review process. I can help explain "
                    "the general process and required information, and if needed I can direct "
                    "you to the appropriate support channel."
                ),
            },
        ],
    },
]


# ── PersonaManager ────────────────────────────────────────────────────────────

class PersonaManager:
    """
    Thin persistence layer for personas.
    All reads go to disk so admin edits are picked up without restart.
    """

    def __init__(self, filepath: str = "personas.json"):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            self._save(DEFAULT_PERSONAS)
            print(f"✓ Seeded {len(DEFAULT_PERSONAS)} default personas → {self.filepath}")

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load(self) -> List[Dict]:
        try:
            data = json.loads(self.filepath.read_text(encoding="utf-8"))
            return data if data else DEFAULT_PERSONAS
        except Exception as e:
            print(f"[WARN] Could not read personas.json: {e}. Using defaults.")
            return DEFAULT_PERSONAS

    def _save(self, personas: List[Dict]) -> None:
        self.filepath.write_text(
            json.dumps(personas, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get_all(self) -> List[Dict]:
        return self.load()

    def get_by_id(self, persona_id: str) -> Optional[Dict]:
        return next((p for p in self.load() if p["id"] == persona_id), None)

    def create(self, data: Dict) -> Dict:
        personas = self.load()
        data = dict(data)
        data["id"] = str(uuid.uuid4())
        personas.append(data)
        self._save(personas)
        return data

    def update(self, persona_id: str, data: Dict) -> Optional[Dict]:
        personas = self.load()
        for i, p in enumerate(personas):
            if p["id"] == persona_id:
                updated = dict(data)
                updated["id"] = persona_id          # id is immutable
                personas[i] = updated
                self._save(personas)
                return updated
        return None

    def delete(self, persona_id: str) -> bool:
        personas = self.load()
        filtered = [p for p in personas if p["id"] != persona_id]
        if len(filtered) == len(personas):
            return False
        self._save(filtered)
        return True

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def valid_ids(self) -> List[str]:
        return [p["id"] for p in self.load()]

    def build_classification_prompt(self) -> str:
        """
        Dynamically generate the intent-classification system prompt from
        whatever personas are currently stored in personas.json.
        """
        personas = self.load()
        bullet_lines = "\n".join(
            f"- {p['id']:<18}: {p['trigger_description']}"
            for p in personas
        )
        id_list = " | ".join(p["id"] for p in personas)
        return (
            "You are an intent classifier for a Tawuniya Insurance chatbot.\n\n"
            "Classify the user's message into EXACTLY ONE of these intents:\n"
            f"{bullet_lines}\n\n"
            f'Output ONLY a JSON object: {{"intent": "<one of: {id_list}>"}}\n'
            "No explanation, no extra text."
        )

    def build_mode_block(self, persona_id: str) -> str:
        """
        Build the few-shot block for a given persona_id to prepend
        to the RAG system prompt.
        """
        persona = self.get_by_id(persona_id)
        if not persona:
            # Graceful fallback: empty block
            return ""

        examples_text = ""
        for i, ex in enumerate(persona.get("examples", []), 1):
            examples_text += (
                f"### Example {i}\n"
                f"User: {ex['user']}\n"
                f"Assistant: {ex['assistant']}\n\n"
            )

        return (
            f"## Reply Mode: {persona['label']}\n"
            f"{persona['instruction']}\n\n"
            f"{examples_text.rstrip()}\n"
        )
