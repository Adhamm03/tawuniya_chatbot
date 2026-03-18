# import re
# import os
# import glob
# import numpy as np
# import faiss
# from dataclasses import dataclass
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer
# from rank_bm25 import BM25Okapi
# from openai import OpenAI
# import json


# @dataclass
# class Chunk:
#     content: str
#     metadata: Dict


# def extract_file_metadata(text: str) -> Dict:
#     """
#     Extract structured metadata from the document header block:
#         [المستند N]
#         العنوان: ...
#         التصنيف: ...
#         الرابط: ...
#         الوصف: ...
#     """
#     metadata = {}

#     doc_id_match = re.search(r'\[المستند\s*(\d+)\]', text)
#     if doc_id_match:
#         metadata['doc_id'] = doc_id_match.group(1)

#     for arabic_key, eng_key in [
#         ('العنوان', 'title'),
#         ('التصنيف', 'category'),
#         ('الرابط', 'url'),
#         ('الوصف', 'description'),
#     ]:
#         match = re.search(rf'{arabic_key}:\s*(.+)', text)
#         if match:
#             metadata[eng_key] = match.group(1).strip()

#     return metadata


# class RAGEngine:
#     def __init__(
#         self,
#         chunks_folder: str,
#         openai_api_key: str,
#         glob_pattern: str = '*.txt',
#     ):
#         """
#         Initialize RAG engine over a folder of pre-split chunk files.
#         Each .txt file is treated as one chunk — no further splitting is done.

#         Args:
#             chunks_folder:  Path to the folder containing the .txt chunk files.
#             openai_api_key: OpenAI API key for GPT-4o-mini.
#             glob_pattern:   File pattern to match inside chunks_folder.
#         """
#         print(f"Loading chunks from: {chunks_folder}")
#         txt_files = sorted(glob.glob(os.path.join(chunks_folder, glob_pattern)))
#         if not txt_files:
#             raise FileNotFoundError(f"No files matching '{glob_pattern}' in {chunks_folder}")
#         print(f"  Found {len(txt_files)} chunk file(s).")

#         self.chunks: List[Chunk] = []
#         for path in txt_files:
#             with open(path, 'r', encoding='utf-8') as f:
#                 content = f.read().strip()

#             meta = extract_file_metadata(content)
#             meta['source_file'] = os.path.basename(path)
#             meta['chunk_size'] = len(content)

#             self.chunks.append(Chunk(content=content, metadata=meta))

#         print(f"  Total chunks: {len(self.chunks)}")

#         # Embeddings
#         print("Creating embeddings...")
#         self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")
#         passage_texts = [f"passage: {c.content}" for c in self.chunks]
#         embeddings = self.embedder.encode(
#             passage_texts, normalize_embeddings=True, show_progress_bar=True
#         )

#         dim = embeddings.shape[1]
#         self.faiss_index = faiss.IndexFlatIP(dim)
#         self.faiss_index.add(np.array(embeddings, dtype=np.float32))
#         self.id2chunk = {i: c for i, c in enumerate(self.chunks)}

#         # BM25
#         print("Initializing BM25...")
#         tokenized_corpus = [c.content.split() for c in self.chunks]
#         self.bm25 = BM25Okapi(tokenized_corpus)

#         # OpenAI client
#         self.client = OpenAI(api_key=openai_api_key)

#         # Conversation memory
#         self.conversation_history: List[Dict] = []
#         self.max_history_turns = 6

#         print("✓ RAG engine ready!")

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------

#     def detect_language(self, text: str) -> str:
#         arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
#         total_chars  = len(re.sub(r'\s', '', text))
#         if total_chars == 0:
#             return 'en'
#         return 'ar' if (arabic_chars / total_chars) > 0.3 else 'en'

#     def translate_to_arabic(self, text: str) -> str:
#         response = self.client.chat.completions.create(
#             model='gpt-4o-mini',
#             messages=[
#                 {'role': 'system',
#                  'content': 'You are a translator. Translate the following text to Arabic. '
#                              'Output ONLY the Arabic translation, no explanations.'},
#                 {'role': 'user', 'content': text},
#             ],
#             temperature=0,
#             max_tokens=1000,
#         )
#         return response.choices[0].message.content.strip()

#     def rewrite_query_with_history(self, query: str) -> str:
#         """Rewrite a follow-up question as a standalone query using conversation history."""
#         if not self.conversation_history:
#             return query

#         history_text = ''
#         for turn in self.conversation_history[-self.max_history_turns:]:
#             role = 'User' if turn['role'] == 'user' else 'Assistant'
#             history_text += f"{role}: {turn['content']}\n"

#         response = self.client.chat.completions.create(
#             model='gpt-4o-mini',
#             messages=[
#                 {'role': 'system',
#                  'content': (
#                      'You are a query rewriter. Given a conversation history and a follow-up '
#                      'question, rewrite the follow-up question as a single, fully self-contained '
#                      'question that includes all necessary context from the history. '
#                      'Output ONLY the rewritten question, nothing else. '
#                      'Keep the same language as the follow-up question.'
#                  )},
#                 {'role': 'user',
#                  'content': (
#                      f'Conversation history:\n{history_text}\n'
#                      f'Follow-up question: {query}\n\nRewritten standalone question:'
#                  )},
#             ],
#             temperature=0,
#             max_tokens=200,
#         )
#         return response.choices[0].message.content.strip()

#     def clear_history(self):
#         """Clear conversation history to start a fresh session."""
#         self.conversation_history = []
#         print("🗑️ Conversation history cleared.")

#     # ------------------------------------------------------------------
#     # Retrieval — Reciprocal Rank Fusion (FAISS + BM25)
#     # ------------------------------------------------------------------

#     def retrieve_chunks(self, query: str, top_k: int = 5, k: int = 60) -> List[Chunk]:
#         """
#         Hybrid retrieval using Reciprocal Rank Fusion over FAISS (semantic)
#         and BM25 (lexical) rankings.

#         RRF score: Σ  1 / (k + rank(d))
#         """
#         q_emb = self.embedder.encode([f'query: {query}'], normalize_embeddings=True)
#         search_k = min(top_k * 2, len(self.chunks))

#         faiss_scores, faiss_ids = self.faiss_index.search(
#             np.array(q_emb, dtype=np.float32), search_k
#         )
#         faiss_ids_list = [int(idx) for idx in faiss_ids[0] if idx >= 0]

#         bm25_scores = self.bm25.get_scores(query.split())
#         bm25_ranking = sorted(
#             range(len(bm25_scores)),
#             key=lambda i: bm25_scores[i],
#             reverse=True,
#         )[:search_k]

#         rrf_scores: Dict[int, float] = {}
#         for rank, idx in enumerate(faiss_ids_list):
#             if idx in self.id2chunk:
#                 rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank + 1)
#         for rank, idx in enumerate(bm25_ranking):
#             if idx in self.id2chunk:
#                 rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank + 1)

#         if not rrf_scores:
#             return []

#         sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
#         return [self.id2chunk[idx] for idx, _ in sorted_chunks[:top_k]
#                 if idx in self.id2chunk]

#     # ------------------------------------------------------------------
#     # Answer generation
#     # ------------------------------------------------------------------

#     def answer_question(
#         self,
#         query: str,
#         debug: bool = False,
#         return_context: bool = False,
#     ):
#         """Answer a question using RAG with conversation memory."""
#         original_lang = self.detect_language(query)

#         standalone_query = self.rewrite_query_with_history(query)
#         if debug and standalone_query != query:
#             print(f'[DEBUG] Original:  {query}')
#             print(f'[DEBUG] Rewritten: {standalone_query}')

#         if original_lang == 'en':
#             retrieval_query = self.translate_to_arabic(standalone_query)
#             if debug:
#                 print(f'[DEBUG] Translated for retrieval: {retrieval_query}')
#         else:
#             retrieval_query = standalone_query

#         retrieved = self.retrieve_chunks(retrieval_query, top_k=10)

#         if not retrieved:
#             no_result = (
#                 'عذراً، ما لقيت معلومات تخص سؤالك.'
#                 if original_lang == 'ar'
#                 else 'Sorry, I could not find relevant information for your question.'
#             )
#             return (no_result, '') if return_context else no_result

#         if debug:
#             print(f'\n[DEBUG] Retrieved {len(retrieved)} chunks:')
#             for i, c in enumerate(retrieved[:5], 1):
#                 print(f"  {i}. [{c.metadata.get('source_file', '')}] "
#                       f"{c.metadata.get('title', 'N/A')} "
#                       f"({c.metadata['chunk_size']} chars)")

#         context_parts = []
#         for c in retrieved:
#             label = c.metadata.get('title') or c.metadata.get('source_file', '')
#             context_parts.append(f'[{label}]\n{c.content}')
#         context = '\n\n'.join(context_parts)

#         if original_lang == 'ar':
#             system_msg = """أنت مساعد متخصص في منتجات وخدمات التأمين لشركة التعاونية للتأمين.
# مهمتك استخراج جميع المعلومات ذات الصلة من النصوص المقدمة فقط.
# لا تضف أي معلومات من خارج النص.

# يجب عليك:
# - الإجابة باللهجة السعودية العامية دائماً (مثل: وش، كيف، إيش، عشان، بس، زين، الحين، إن شاء الله، ما عندي، تقدر، ودّك).
# - استخراج جميع العناصر المذكورة في النص دون حذف أي منها.
# - عدم تكرار أي بند.
# - عرض الإجابة في نقاط واضحة إذا كانت موجودة في النص."""

#             user_prompt = f"""النصوص المرجعية:

# {context}

# السؤال:
# {query}

# أجب الآن باللهجة السعودية."""

#         else:
#             system_msg = """You are a professional insurance assistant specializing in Tawuniya Insurance products.
# Your task is to extract ALL relevant information strictly from the provided texts and answer in English.
# Do NOT add information from outside the texts.

# CRITICAL LANGUAGE RULE:
# - YOUR ENTIRE ANSWER MUST BE IN ENGLISH. Do not include any Arabic text in your response.
# - Translate all relevant content from Arabic to English completely.

# If the question requires multiple items, list ALL of them without omission or repetition."""

#             user_prompt = f"""Source texts:

# {context}

# Question (answer in English only):
# {query}

# Provide your answer now in English."""

#         response = self.client.chat.completions.create(
#             model='gpt-4o-mini',
#             messages=[
#                 {'role': 'system', 'content': system_msg},
#                 {'role': 'user',   'content': user_prompt},
#             ],
#             temperature=0.3,
#             max_tokens=1500,
#         )
#         answer = response.choices[0].message.content

#         self.conversation_history.append({'role': 'user',      'content': standalone_query})
#         self.conversation_history.append({'role': 'assistant', 'content': answer})
#         if len(self.conversation_history) > self.max_history_turns:
#             self.conversation_history = self.conversation_history[-self.max_history_turns:]

#         return (answer, context) if return_context else answer

#     # ------------------------------------------------------------------
#     # Follow-up question generation
#     # ------------------------------------------------------------------

#     def generate_followup_questions(
#         self,
#         query: str,
#         answer: str,
#         context: str,
#         lang: str,
#     ) -> List[str]:
#         """Generate 3 context-bounded follow-up questions."""
#         lang_name = 'Arabic' if lang == 'ar' else 'English'
#         system_prompt = (
#             f'You are an assistant helping users navigate Tawuniya Insurance information. '
#             f'Based on the user\'s question, the answer, and ONLY the following texts, '
#             f'suggest exactly 3 short follow-up questions.\n'
#             f'CRITICAL: questions must be answerable from the provided texts only.\n'
#             f'Output JSON: {{"questions": ["Q1?", "Q2?", "Q3?"]}}\n'
#             f'Questions MUST be in {lang_name}.\n\n'
#             f'Texts:\n{context}'
#         )
#         user_prompt = (
#             f'User Question: {query}\nChatbot Answer: {answer}\n\n'
#             f'Provide the 3 follow-up questions as JSON.'
#         )
#         try:
#             response = self.client.chat.completions.create(
#                 model='gpt-4o-mini',
#                 messages=[
#                     {'role': 'system', 'content': system_prompt},
#                     {'role': 'user',   'content': user_prompt},
#                 ],
#                 temperature=0.3,
#                 max_tokens=200,
#                 response_format={'type': 'json_object'},
#             )
#             result = json.loads(response.choices[0].message.content.strip())
#             return result.get('questions', [])
#         except Exception as e:
#             print(f'Follow-up generation failed: {e}')
#             return []

#     # ------------------------------------------------------------------
#     # Statistics
#     # ------------------------------------------------------------------

#     def get_statistics(self) -> Dict:
#         sizes = [c.metadata['chunk_size'] for c in self.chunks]
#         return {
#             'total_chunks':   len(self.chunks),
#             'avg_chunk_size': round(sum(sizes) / len(sizes), 1) if sizes else 0,
#             'max_chunk_size': max(sizes) if sizes else 0,
#             'min_chunk_size': min(sizes) if sizes else 0,
#         }


# # ========== ENTRY POINT ==========

# if __name__ == '__main__':
#     CHUNKS_FOLDER = os.path.join('splitted_chunks', 'splitted_chunks')
#     OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

#     engine = RAGEngine(
#         chunks_folder=CHUNKS_FOLDER,
#         openai_api_key=OPENAI_API_KEY,
#     )

#     stats = engine.get_statistics()
#     print('\n' + '=' * 70)
#     print('STATISTICS')
#     print('=' * 70)
#     for k, v in stats.items():
#         print(f'  {k}: {v}')

#     print('\n' + '=' * 70)
#     print('EXAMPLE QUERIES')
#     print('=' * 70)

#     print('\n1. Arabic query:')
#     answer = engine.answer_question('ما هي مزايا برنامج تأمين العمرة؟', debug=True)
#     print(f'\nAnswer:\n{answer}')

#     print('\n\n2. English query:')
#     answer = engine.answer_question('What does medical malpractice insurance cover?', debug=True)
#     print(f'\nAnswer:\n{answer}')
########################################################################################################################
import re
import os
import glob
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI
import json


@dataclass
class Chunk:
    content: str
    metadata: Dict


# ══════════════════════════════════════════════════════════════════════════════
# Few-shot reply mode definitions
# Each mode has: system_instruction + one-shot example (user/assistant pair)
# These are injected into the prompt BEFORE the RAG context so the model
# learns the tone, structure, and boundaries expected for that intent type.
# ══════════════════════════════════════════════════════════════════════════════

REPLY_MODES = {
    "knowledge": {
        "label": "Knowledge mode — explain insurance term or concept",
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
                    "more cost-efficient option with a focused network. The exact benefits for "
                    "each class are defined in your policy schedule. Would you like me to "
                    "explain what a coverage limit means, or how the network levels work?"
                ),
            },
        ],
    },
    "consultant": {
        "label": "Consultant mode — recommend or guide based on customer situation",
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
                    "share these details, I can guide you toward the most suitable option and "
                    "explain why."
                ),
            },
            {
                "user": "I have 18 employees in Riyadh, what package fits me?",
                "assistant": (
                    "Good, that gives me a useful starting point. For a team of 18 in Riyadh, "
                    "there are a few options worth considering depending on your budget and the "
                    "level of network coverage you need. To narrow it down further, it would "
                    "help to know your approximate budget per employee and whether you need "
                    "coverage for dependents as well. Once I have that, I can point you to the "
                    "most suitable direction."
                ),
            },
        ],
    },
    "competitor": {
        "label": "Competitor mode — handle comparison or competitor questions",
        "instruction": (
            "You are in COMPETITOR MODE. Never speak negatively about other insurers. "
            "Stay objective. Help the customer identify what matters most to them, then "
            "highlight where Tawuniya fits those priorities."
        ),
        "examples": [
            {
                "user": "Why should I choose Tawuniya over another insurer?",
                "assistant": (
                    "The right choice depends on your business priorities. Many customers look "
                    "at factors such as provider network, service quality, digital experience, "
                    "claims support, and budget fit. Tawuniya's value is often strongest when "
                    "customers are looking for a trusted insurer with strong market experience "
                    "and flexible options. If you tell me what matters most to your company, "
                    "I can help you assess the best fit objectively."
                ),
            },
            {
                "user": "Is Tawuniya better than Bupa?",
                "assistant": (
                    "That depends on what your company values most. Both are established "
                    "insurers operating in the Saudi market, and each has strengths in "
                    "different areas. Rather than a simple comparison, the better question is "
                    "which one fits your specific needs — in terms of network coverage, "
                    "service channels, plan flexibility, and budget. If you share your "
                    "priorities, I can help you think through the decision more clearly."
                ),
            },
        ],
    },
    "objection": {
        "label": "Objection handling mode — price or value objections",
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
            {
                "user": "Your competitor is cheaper.",
                "assistant": (
                    "I understand, and price is always a fair point to raise. The difference "
                    "in cost often reflects differences in network size, coverage depth, "
                    "service responsiveness, or claims handling. Before deciding on price "
                    "alone, it may be worth comparing what is actually included. If you share "
                    "what the competing offer covers, I can help you assess whether the "
                    "comparison is truly like-for-like."
                ),
            },
        ],
    },
    "complaint": {
        "label": "Complaint handling mode — frustrated or confused user",
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
                    "from there. If this requires case-specific support, I can also direct "
                    "you to the right next step."
                ),
            },
        ],
    },
    "sales": {
        "label": "Sales progression mode — lead capture or quote request",
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
                    "renewal. Once I have that, I can help narrow the suitable option and "
                    "prepare the next step."
                ),
            },
            {
                "user": "I want a quote today.",
                "assistant": (
                    "Of course, let's get that moving. I just need four quick details: how "
                    "many employees you have, which city you are based in, your industry, and "
                    "whether this is a new policy or a renewal. Share those and I will guide "
                    "you to the right next step right away."
                ),
            },
        ],
    },
    "restricted": {
        "label": "Restricted mode — guarantee requests or out-of-scope commitments",
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
                    "you to the appropriate support channel for a case-specific review."
                ),
            },
            {
                "user": "Can you guarantee approval?",
                "assistant": (
                    "Guaranteeing approval is not something I am able to do — that decision "
                    "follows a formal review process based on your specific policy terms and "
                    "case details. What I can do is walk you through how the process works, "
                    "what documents are typically needed, and where to go if you need a "
                    "case-specific answer from the right team."
                ),
            },
        ],
    },
}

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a Tawuniya Insurance chatbot.

Classify the user's message into EXACTLY ONE of these 7 intents:
- knowledge     : asking about an insurance term, concept, or product feature
- consultant    : asking for a recommendation, plan suggestion, or what fits their situation
- competitor    : comparing Tawuniya to another insurer, or asking why choose Tawuniya
- objection     : expressing concern about price, value, or hesitation about a product
- complaint     : frustrated, confused, expressing dissatisfaction, or saying no one helped
- sales         : requesting a quote, asking how to buy, or showing purchase intent
- restricted    : requesting a guarantee, approval confirmation, or out-of-scope commitment

Output ONLY a JSON object: {"intent": "<one of the 7 intents above>"}
No explanation, no extra text."""


def extract_file_metadata(text: str) -> Dict:
    """
    Extract structured metadata from the document header block:
        [المستند N]
        العنوان: ...
        التصنيف: ...
        الرابط: ...
        الوصف: ...
    """
    metadata = {}

    doc_id_match = re.search(r'\[المستند\s*(\d+)\]', text)
    if doc_id_match:
        metadata['doc_id'] = doc_id_match.group(1)

    for arabic_key, eng_key in [
        ('العنوان', 'title'),
        ('التصنيف', 'category'),
        ('الرابط', 'url'),
        ('الوصف', 'description'),
    ]:
        match = re.search(rf'{arabic_key}:\s*(.+)', text)
        if match:
            metadata[eng_key] = match.group(1).strip()

    return metadata


class RAGEngine:
    def __init__(
        self,
        chunks_folder: str,
        openai_api_key: str,
        glob_pattern: str = '*.txt',
    ):
        """
        Initialize RAG engine over a folder of pre-split chunk files.
        Each .txt file is treated as one chunk — no further splitting is done.

        Args:
            chunks_folder:  Path to the folder containing the .txt chunk files.
            openai_api_key: OpenAI API key for GPT-4o-mini.
            glob_pattern:   File pattern to match inside chunks_folder.
        """
        print(f"Loading chunks from: {chunks_folder}")
        txt_files = sorted(glob.glob(os.path.join(chunks_folder, glob_pattern)))
        if not txt_files:
            raise FileNotFoundError(f"No files matching '{glob_pattern}' in {chunks_folder}")
        print(f"  Found {len(txt_files)} chunk file(s).")

        self.chunks: List[Chunk] = []
        for path in txt_files:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            meta = extract_file_metadata(content)
            meta['source_file'] = os.path.basename(path)
            meta['chunk_size'] = len(content)

            self.chunks.append(Chunk(content=content, metadata=meta))

        print(f"  Total chunks: {len(self.chunks)}")

        # Embeddings
        print("Creating embeddings...")
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")
        passage_texts = [f"passage: {c.content}" for c in self.chunks]
        embeddings = self.embedder.encode(
            passage_texts, normalize_embeddings=True, show_progress_bar=True
        )

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))
        self.id2chunk = {i: c for i, c in enumerate(self.chunks)}

        # BM25
        print("Initializing BM25...")
        tokenized_corpus = [c.content.split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # OpenAI client
        self.client = OpenAI(api_key=openai_api_key)

        # Conversation memory
        self.conversation_history: List[Dict] = []
        self.max_history_turns = 6

        print("✓ RAG engine ready!")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def detect_language(self, text: str) -> str:
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars  = len(re.sub(r'\s', '', text))
        if total_chars == 0:
            return 'en'
        return 'ar' if (arabic_chars / total_chars) > 0.3 else 'en'

    def translate_to_arabic(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system',
                 'content': 'You are a translator. Translate the following text to Arabic. '
                             'Output ONLY the Arabic translation, no explanations.'},
                {'role': 'user', 'content': text},
            ],
            temperature=0,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()

    def rewrite_query_with_history(self, query: str) -> str:
        """Rewrite a follow-up question as a standalone query using conversation history."""
        if not self.conversation_history:
            return query

        history_text = ''
        for turn in self.conversation_history[-self.max_history_turns:]:
            role = 'User' if turn['role'] == 'user' else 'Assistant'
            history_text += f"{role}: {turn['content']}\n"

        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system',
                 'content': (
                     'You are a query rewriter. Given a conversation history and a follow-up '
                     'question, rewrite the follow-up question as a single, fully self-contained '
                     'question that includes all necessary context from the history. '
                     'Output ONLY the rewritten question, nothing else. '
                     'Keep the same language as the follow-up question.'
                 )},
                {'role': 'user',
                 'content': (
                     f'Conversation history:\n{history_text}\n'
                     f'Follow-up question: {query}\n\nRewritten standalone question:'
                 )},
            ],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    def clear_history(self):
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared.")

    # ------------------------------------------------------------------
    # NEW: Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> str:
        """
        Classify the user query into one of 7 reply modes using GPT-4o-mini.
        Returns one of: knowledge | consultant | competitor | objection |
                        complaint | sales | restricted
        Falls back to 'knowledge' on any error.
        """
        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': INTENT_CLASSIFICATION_PROMPT},
                    {'role': 'user',   'content': query},
                ],
                temperature=0,
                max_tokens=30,
                response_format={'type': 'json_object'},
            )
            result = json.loads(response.choices[0].message.content.strip())
            intent = result.get('intent', 'knowledge')
            if intent not in REPLY_MODES:
                intent = 'knowledge'
            return intent
        except Exception as e:
            print(f'[WARN] Intent classification failed: {e}')
            return 'knowledge'

    def _build_mode_block(self, intent: str, lang: str) -> str:
        """
        Build the few-shot mode block to prepend to the system prompt.
        Renders ALL examples for the detected intent so the model learns
        the exact reply pattern from multiple real samples.
        """
        mode = REPLY_MODES[intent]
        examples_text = ""
        for i, ex in enumerate(mode["examples"], 1):
            examples_text += (
                f"### Example {i}\n"
                f"User: {ex['user']}\n"
                f"Assistant: {ex['assistant']}\n\n"
            )
        block = (
            f"## Reply Mode: {mode['label']}\n"
            f"{mode['instruction']}\n\n"
            f"{examples_text.rstrip()}\n"
        )
        return block

    # ------------------------------------------------------------------
    # Retrieval — Reciprocal Rank Fusion (FAISS + BM25)
    # ------------------------------------------------------------------

    def retrieve_chunks(self, query: str, top_k: int = 5, k: int = 60) -> List[Chunk]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion over FAISS (semantic)
        and BM25 (lexical) rankings.

        RRF score: Σ  1 / (k + rank(d))
        """
        q_emb = self.embedder.encode([f'query: {query}'], normalize_embeddings=True)
        search_k = min(top_k * 2, len(self.chunks))

        faiss_scores, faiss_ids = self.faiss_index.search(
            np.array(q_emb, dtype=np.float32), search_k
        )
        faiss_ids_list = [int(idx) for idx in faiss_ids[0] if idx >= 0]

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranking = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:search_k]

        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(faiss_ids_list):
            if idx in self.id2chunk:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            if idx in self.id2chunk:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank + 1)

        if not rrf_scores:
            return []

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.id2chunk[idx] for idx, _ in sorted_chunks[:top_k]
                if idx in self.id2chunk]

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    # Modes that need document retrieval from the knowledge base
    RAG_MODES           = {"knowledge"}
    # Modes that are conversational — no document retrieval needed
    CONVERSATIONAL_MODES = {"consultant", "competitor", "objection",
                             "complaint", "sales", "restricted"}

    def answer_question(
        self,
        query: str,
        debug: bool = False,
        return_context: bool = False,
    ):
        """Answer a question using RAG with conversation memory and smart reply modes."""
        original_lang = self.detect_language(query)

        # ── Step 1: classify intent ────────────────────────────────────
        intent = self.classify_intent(query)
        if debug:
            print(f'[DEBUG] Intent: {intent}')

        # ── Step 2: rewrite query ──────────────────────────────────────
        standalone_query = self.rewrite_query_with_history(query)
        if debug and standalone_query != query:
            print(f'[DEBUG] Original:  {query}')
            print(f'[DEBUG] Rewritten: {standalone_query}')

        # ── Step 3: build mode block ───────────────────────────────────
        mode_block = self._build_mode_block(intent, original_lang)

        # ── Step 4: RAG path vs conversational path ────────────────────
        context = ""

        if intent in self.RAG_MODES:
            # ── RAG path: retrieve from knowledge base ─────────────────
            if debug:
                print(f'[DEBUG] Path: RAG — retrieving from knowledge base')

            if original_lang == 'en':
                retrieval_query = self.translate_to_arabic(standalone_query)
                if debug:
                    print(f'[DEBUG] Translated for retrieval: {retrieval_query}')
            else:
                retrieval_query = standalone_query

            retrieved = self.retrieve_chunks(retrieval_query, top_k=10)

            if not retrieved:
                no_result = (
                    'عذراً، ما لقيت معلومات تخص سؤالك.'
                    if original_lang == 'ar'
                    else 'Sorry, I could not find relevant information for your question.'
                )
                return (no_result, '') if return_context else no_result

            if debug:
                print(f'\n[DEBUG] Retrieved {len(retrieved)} chunks:')
                for i, c in enumerate(retrieved[:5], 1):
                    print(f"  {i}. [{c.metadata.get('source_file', '')}] "
                          f"{c.metadata.get('title', 'N/A')} "
                          f"({c.metadata['chunk_size']} chars)")

            context_parts = []
            for c in retrieved:
                label = c.metadata.get('title') or c.metadata.get('source_file', '')
                context_parts.append(f'[{label}]\n{c.content}')
            context = '\n\n'.join(context_parts)

            # Build RAG-grounded system prompt
            if original_lang == 'ar':
                system_msg = f"""{mode_block}

أنت مساعد متخصص في منتجات وخدمات التأمين لشركة التعاونية للتأمين.
مهمتك استخراج جميع المعلومات ذات الصلة من النصوص المقدمة فقط.
لا تضف أي معلومات من خارج النص.

يجب عليك:
- الإجابة باللهجة السعودية العامية دائماً (مثل: وش، كيف، إيش، عشان، بس، زين، الحين، إن شاء الله، ما عندي، تقدر، ودّك).
- استخراج جميع العناصر المذكورة في النص دون حذف أي منها.
- عدم تكرار أي بند.
- عرض الإجابة في نقاط واضحة إذا كانت موجودة في النص."""

                user_prompt = f"""النصوص المرجعية:

{context}

السؤال:
{query}

أجب الآن باللهجة السعودية."""

            else:
                system_msg = f"""{mode_block}

You are a professional insurance assistant specializing in Tawuniya Insurance products.
Your task is to extract ALL relevant information strictly from the provided texts and answer in English.
Do NOT add information from outside the texts.

CRITICAL LANGUAGE RULE:
- YOUR ENTIRE ANSWER MUST BE IN ENGLISH. Do not include any Arabic text in your response.
- Translate all relevant content from Arabic to English completely.

If the question requires multiple items, list ALL of them without omission or repetition."""

                user_prompt = f"""Source texts:

{context}

Question (answer in English only):
{query}

Provide your answer now in English."""

        else:
            # ── Conversational path: no retrieval, mode instructions only ──
            if debug:
                print(f'[DEBUG] Path: Conversational — no retrieval, mode instructions only')

            if original_lang == 'ar':
                system_msg = f"""{mode_block}

أنت مساعد متخصص في منتجات وخدمات التأمين لشركة التعاونية للتأمين.
الإجابة باللهجة السعودية العامية دائماً (مثل: وش، كيف، إيش، عشان، بس، زين، الحين، إن شاء الله، ما عندي، تقدر، ودّك).
أجب بناءً على أسلوب الرد المحدد أعلاه فقط، دون الاستناد لأي نصوص خارجية."""

                user_prompt = f"""{query}"""

            else:
                system_msg = f"""{mode_block}

You are a professional insurance assistant specializing in Tawuniya Insurance products.
Reply based on the mode instructions and examples above only.
YOUR ENTIRE ANSWER MUST BE IN ENGLISH."""

                user_prompt = f"""{query}"""

        # ── Step 5: generate answer ────────────────────────────────────
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user',   'content': user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        answer = response.choices[0].message.content

        # ── Step 6: update history ─────────────────────────────────────
        self.conversation_history.append({'role': 'user',      'content': standalone_query})
        self.conversation_history.append({'role': 'assistant', 'content': answer})
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

        return (answer, context) if return_context else answer

    # ------------------------------------------------------------------
    # Follow-up question generation
    # ------------------------------------------------------------------

    def generate_followup_questions(
        self,
        query: str,
        answer: str,
        context: str,
        lang: str,
    ) -> List[str]:
        """Generate 3 context-bounded follow-up questions."""
        lang_name = 'Arabic' if lang == 'ar' else 'English'
        system_prompt = (
            f'You are an assistant helping users navigate Tawuniya Insurance information. '
            f'Based on the user\'s question, the answer, and ONLY the following texts, '
            f'suggest exactly 3 short follow-up questions.\n'
            f'CRITICAL: questions must be answerable from the provided texts only.\n'
            f'Output JSON: {{"questions": ["Q1?", "Q2?", "Q3?"]}}\n'
            f'Questions MUST be in {lang_name}.\n\n'
            f'Texts:\n{context}'
        )
        user_prompt = (
            f'User Question: {query}\nChatbot Answer: {answer}\n\n'
            f'Provide the 3 follow-up questions as JSON.'
        )
        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt},
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={'type': 'json_object'},
            )
            result = json.loads(response.choices[0].message.content.strip())
            return result.get('questions', [])
        except Exception as e:
            print(f'Follow-up generation failed: {e}')
            return []

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        sizes = [c.metadata['chunk_size'] for c in self.chunks]
        return {
            'total_chunks':   len(self.chunks),
            'avg_chunk_size': round(sum(sizes) / len(sizes), 1) if sizes else 0,
            'max_chunk_size': max(sizes) if sizes else 0,
            'min_chunk_size': min(sizes) if sizes else 0,
        }


# ========== ENTRY POINT ==========

if __name__ == '__main__':
    CHUNKS_FOLDER = os.path.join('splitted_chunks', 'splitted_chunks')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

    engine = RAGEngine(
        chunks_folder=CHUNKS_FOLDER,
        openai_api_key=OPENAI_API_KEY,
    )

    stats = engine.get_statistics()
    print('\n' + '=' * 70)
    print('STATISTICS')
    print('=' * 70)
    for k, v in stats.items():
        print(f'  {k}: {v}')

    print('\n' + '=' * 70)
    print('EXAMPLE QUERIES')
    print('=' * 70)

    print('\n1. Arabic query:')
    answer = engine.answer_question('ما هي مزايا برنامج تأمين العمرة؟', debug=True)
    print(f'\nAnswer:\n{answer}')

    print('\n\n2. English query:')
    answer = engine.answer_question('What does medical malpractice insurance cover?', debug=True)
    print(f'\nAnswer:\n{answer}')

##########################################################################################

import re
import os
import glob
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI
import json
 
 
@dataclass
class Chunk:
    content: str
    metadata: Dict
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Few-shot reply mode definitions
# Each mode has: system_instruction + one-shot example (user/assistant pair)
# These are injected into the prompt BEFORE the RAG context so the model
# learns the tone, structure, and boundaries expected for that intent type.
# ══════════════════════════════════════════════════════════════════════════════
 
REPLY_MODES = {
    "knowledge": {
        "label": "Knowledge mode — explain insurance term or concept",
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
                    "more cost-efficient option with a focused network. The exact benefits for "
                    "each class are defined in your policy schedule. Would you like me to "
                    "explain what a coverage limit means, or how the network levels work?"
                ),
            },
        ],
    },
    "consultant": {
        "label": "Consultant mode — recommend or guide based on customer situation",
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
                    "share these details, I can guide you toward the most suitable option and "
                    "explain why."
                ),
            },
            {
                "user": "I have 18 employees in Riyadh, what package fits me?",
                "assistant": (
                    "Good, that gives me a useful starting point. For a team of 18 in Riyadh, "
                    "there are a few options worth considering depending on your budget and the "
                    "level of network coverage you need. To narrow it down further, it would "
                    "help to know your approximate budget per employee and whether you need "
                    "coverage for dependents as well. Once I have that, I can point you to the "
                    "most suitable direction."
                ),
            },
        ],
    },
    "competitor": {
        "label": "Competitor mode — handle comparison or competitor questions",
        "instruction": (
            "You are in COMPETITOR MODE. Never speak negatively about other insurers. "
            "Stay objective. Help the customer identify what matters most to them, then "
            "highlight where Tawuniya fits those priorities."
        ),
        "examples": [
            {
                "user": "Why should I choose Tawuniya over another insurer?",
                "assistant": (
                    "The right choice depends on your business priorities. Many customers look "
                    "at factors such as provider network, service quality, digital experience, "
                    "claims support, and budget fit. Tawuniya's value is often strongest when "
                    "customers are looking for a trusted insurer with strong market experience "
                    "and flexible options. If you tell me what matters most to your company, "
                    "I can help you assess the best fit objectively."
                ),
            },
            {
                "user": "Is Tawuniya better than Bupa?",
                "assistant": (
                    "That depends on what your company values most. Both are established "
                    "insurers operating in the Saudi market, and each has strengths in "
                    "different areas. Rather than a simple comparison, the better question is "
                    "which one fits your specific needs — in terms of network coverage, "
                    "service channels, plan flexibility, and budget. If you share your "
                    "priorities, I can help you think through the decision more clearly."
                ),
            },
        ],
    },
    "objection": {
        "label": "Objection handling mode — price or value objections",
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
            {
                "user": "Your competitor is cheaper.",
                "assistant": (
                    "I understand, and price is always a fair point to raise. The difference "
                    "in cost often reflects differences in network size, coverage depth, "
                    "service responsiveness, or claims handling. Before deciding on price "
                    "alone, it may be worth comparing what is actually included. If you share "
                    "what the competing offer covers, I can help you assess whether the "
                    "comparison is truly like-for-like."
                ),
            },
        ],
    },
    "complaint": {
        "label": "Complaint handling mode — frustrated or confused user",
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
                    "from there. If this requires case-specific support, I can also direct "
                    "you to the right next step."
                ),
            },
        ],
    },
    "sales": {
        "label": "Sales progression mode — lead capture or quote request",
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
                    "renewal. Once I have that, I can help narrow the suitable option and "
                    "prepare the next step."
                ),
            },
            {
                "user": "I want a quote today.",
                "assistant": (
                    "Of course, let's get that moving. I just need four quick details: how "
                    "many employees you have, which city you are based in, your industry, and "
                    "whether this is a new policy or a renewal. Share those and I will guide "
                    "you to the right next step right away."
                ),
            },
        ],
    },
    "restricted": {
        "label": "Restricted mode — guarantee requests or out-of-scope commitments",
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
                    "you to the appropriate support channel for a case-specific review."
                ),
            },
            {
                "user": "Can you guarantee approval?",
                "assistant": (
                    "Guaranteeing approval is not something I am able to do — that decision "
                    "follows a formal review process based on your specific policy terms and "
                    "case details. What I can do is walk you through how the process works, "
                    "what documents are typically needed, and where to go if you need a "
                    "case-specific answer from the right team."
                ),
            },
        ],
    },
}
 
INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a Tawuniya Insurance chatbot.
 
Classify the user's message into EXACTLY ONE of these 7 intents:
- knowledge     : asking about an insurance term, concept, or product feature
- consultant    : asking for a recommendation, plan suggestion, or what fits their situation
- competitor    : comparing Tawuniya to another insurer, or asking why choose Tawuniya
- objection     : expressing concern about price, value, or hesitation about a product
- complaint     : frustrated, confused, expressing dissatisfaction, or saying no one helped
- sales         : requesting a quote, asking how to buy, or showing purchase intent
- restricted    : requesting a guarantee, approval confirmation, or out-of-scope commitment
 
Output ONLY a JSON object: {"intent": "<one of the 7 intents above>"}
No explanation, no extra text."""
 
 
def extract_file_metadata(text: str) -> Dict:
    """
    Extract structured metadata from the document header block:
        [المستند N]
        العنوان: ...
        التصنيف: ...
        الرابط: ...
        الوصف: ...
    """
    metadata = {}
 
    doc_id_match = re.search(r'\[المستند\s*(\d+)\]', text)
    if doc_id_match:
        metadata['doc_id'] = doc_id_match.group(1)
 
    for arabic_key, eng_key in [
        ('العنوان', 'title'),
        ('التصنيف', 'category'),
        ('الرابط', 'url'),
        ('الوصف', 'description'),
    ]:
        match = re.search(rf'{arabic_key}:\s*(.+)', text)
        if match:
            metadata[eng_key] = match.group(1).strip()
 
    return metadata
 
 
class RAGEngine:
    def __init__(
        self,
        chunks_folder: str,
        openai_api_key: str,
        glob_pattern: str = '*.txt',
    ):
        """
        Initialize RAG engine over a folder of pre-split chunk files.
        Each .txt file is treated as one chunk — no further splitting is done.
 
        Args:
            chunks_folder:  Path to the folder containing the .txt chunk files.
            openai_api_key: OpenAI API key for GPT-4o-mini.
            glob_pattern:   File pattern to match inside chunks_folder.
        """
        print(f"Loading chunks from: {chunks_folder}")
        txt_files = sorted(glob.glob(os.path.join(chunks_folder, glob_pattern)))
        if not txt_files:
            raise FileNotFoundError(f"No files matching '{glob_pattern}' in {chunks_folder}")
        print(f"  Found {len(txt_files)} chunk file(s).")
 
        self.chunks: List[Chunk] = []
        for path in txt_files:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
 
            meta = extract_file_metadata(content)
            meta['source_file'] = os.path.basename(path)
            meta['chunk_size'] = len(content)
 
            self.chunks.append(Chunk(content=content, metadata=meta))
 
        print(f"  Total chunks: {len(self.chunks)}")
 
        # Embeddings
        print("Creating embeddings...")
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")
        passage_texts = [f"passage: {c.content}" for c in self.chunks]
        embeddings = self.embedder.encode(
            passage_texts, normalize_embeddings=True, show_progress_bar=True
        )
 
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))
        self.id2chunk = {i: c for i, c in enumerate(self.chunks)}
 
        # BM25
        print("Initializing BM25...")
        tokenized_corpus = [c.content.split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
 
        # OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
 
        # Conversation memory
        self.conversation_history: List[Dict] = []
        self.max_history_turns = 6
 
        print("✓ RAG engine ready!")
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    def detect_language(self, text: str) -> str:
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        total_chars  = len(re.sub(r'\s', '', text))
        if total_chars == 0:
            return 'en'
        return 'ar' if (arabic_chars / total_chars) > 0.3 else 'en'
 
    def translate_to_arabic(self, text: str) -> str:
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system',
                 'content': 'You are a translator. Translate the following text to Arabic. '
                             'Output ONLY the Arabic translation, no explanations.'},
                {'role': 'user', 'content': text},
            ],
            temperature=0,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
 
    def rewrite_query_with_history(self, query: str) -> str:
        """Rewrite a follow-up question as a standalone query using conversation history."""
        if not self.conversation_history:
            return query
 
        history_text = ''
        for turn in self.conversation_history[-self.max_history_turns:]:
            role = 'User' if turn['role'] == 'user' else 'Assistant'
            history_text += f"{role}: {turn['content']}\n"
 
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system',
                 'content': (
                     'You are a query rewriter. Given a conversation history and a follow-up '
                     'question, rewrite the follow-up question as a single, fully self-contained '
                     'question that includes all necessary context from the history. '
                     'Output ONLY the rewritten question, nothing else. '
                     'Keep the same language as the follow-up question.'
                 )},
                {'role': 'user',
                 'content': (
                     f'Conversation history:\n{history_text}\n'
                     f'Follow-up question: {query}\n\nRewritten standalone question:'
                 )},
            ],
            temperature=0,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
 
    def clear_history(self):
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared.")
 
    # ------------------------------------------------------------------
    # NEW: Intent classification
    # ------------------------------------------------------------------
 
    def classify_intent(self, query: str) -> str:
        """
        Classify the user query into one of 7 reply modes using GPT-4o-mini.
        Returns one of: knowledge | consultant | competitor | objection |
                        complaint | sales | restricted
        Falls back to 'knowledge' on any error.
        """
        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': INTENT_CLASSIFICATION_PROMPT},
                    {'role': 'user',   'content': query},
                ],
                temperature=0,
                max_tokens=30,
                response_format={'type': 'json_object'},
            )
            result = json.loads(response.choices[0].message.content.strip())
            intent = result.get('intent', 'knowledge')
            if intent not in REPLY_MODES:
                intent = 'knowledge'
            return intent
        except Exception as e:
            print(f'[WARN] Intent classification failed: {e}')
            return 'knowledge'
 
    def _build_mode_block(self, intent: str, lang: str) -> str:
        """
        Build the few-shot mode block to prepend to the system prompt.
        Renders ALL examples for the detected intent so the model learns
        the exact reply pattern from multiple real samples.
        """
        mode = REPLY_MODES[intent]
        examples_text = ""
        for i, ex in enumerate(mode["examples"], 1):
            examples_text += (
                f"### Example {i}\n"
                f"User: {ex['user']}\n"
                f"Assistant: {ex['assistant']}\n\n"
            )
        block = (
            f"## Reply Mode: {mode['label']}\n"
            f"{mode['instruction']}\n\n"
            f"{examples_text.rstrip()}\n"
        )
        return block
 
    # ------------------------------------------------------------------
    # Retrieval — Reciprocal Rank Fusion (FAISS + BM25)
    # ------------------------------------------------------------------
 
    def retrieve_chunks(self, query: str, top_k: int = 5, k: int = 60) -> List[Chunk]:
        """
        Hybrid retrieval using Reciprocal Rank Fusion over FAISS (semantic)
        and BM25 (lexical) rankings.
 
        RRF score: Σ  1 / (k + rank(d))
        """
        q_emb = self.embedder.encode([f'query: {query}'], normalize_embeddings=True)
        search_k = min(top_k * 2, len(self.chunks))
 
        faiss_scores, faiss_ids = self.faiss_index.search(
            np.array(q_emb, dtype=np.float32), search_k
        )
        faiss_ids_list = [int(idx) for idx in faiss_ids[0] if idx >= 0]
 
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranking = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:search_k]
 
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(faiss_ids_list):
            if idx in self.id2chunk:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            if idx in self.id2chunk:
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1 / (k + rank + 1)
 
        if not rrf_scores:
            return []
 
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.id2chunk[idx] for idx, _ in sorted_chunks[:top_k]
                if idx in self.id2chunk]
 
    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------
 
    # Modes that need document retrieval from the knowledge base
    RAG_MODES           = {"knowledge"}
    # Modes that are conversational — no document retrieval needed
    CONVERSATIONAL_MODES = {"consultant", "competitor", "objection",
                             "complaint", "sales", "restricted"}
 
    def answer_question(
        self,
        query: str,
        debug: bool = False,
        return_context: bool = False,
    ):
        """Answer a question using RAG with conversation memory and smart reply modes."""
        original_lang = self.detect_language(query)
 
        # ── Step 1: classify intent ────────────────────────────────────
        intent = self.classify_intent(query)
        if debug:
            print(f'[DEBUG] Intent: {intent}')
 
        # ── Step 2: rewrite query ──────────────────────────────────────
        standalone_query = self.rewrite_query_with_history(query)
        if debug and standalone_query != query:
            print(f'[DEBUG] Original:  {query}')
            print(f'[DEBUG] Rewritten: {standalone_query}')
 
        # ── Step 3: build mode block ───────────────────────────────────
        mode_block = self._build_mode_block(intent, original_lang)
 
        # ── Step 4: RAG path vs conversational path ────────────────────
        context = ""
 
        if intent in self.RAG_MODES:
            # ── RAG path: retrieve from knowledge base ─────────────────
            if debug:
                print(f'[DEBUG] Path: RAG — retrieving from knowledge base')
 
            if original_lang == 'en':
                retrieval_query = self.translate_to_arabic(standalone_query)
                if debug:
                    print(f'[DEBUG] Translated for retrieval: {retrieval_query}')
            else:
                retrieval_query = standalone_query
 
            retrieved = self.retrieve_chunks(retrieval_query, top_k=10)
 
            if not retrieved:
                no_result = (
                    'عذراً، ما لقيت معلومات تخص سؤالك.'
                    if original_lang == 'ar'
                    else 'Sorry, I could not find relevant information for your question.'
                )
                return (no_result, '') if return_context else no_result
 
            if debug:
                print(f'\n[DEBUG] Retrieved {len(retrieved)} chunks:')
                for i, c in enumerate(retrieved[:5], 1):
                    print(f"  {i}. [{c.metadata.get('source_file', '')}] "
                          f"{c.metadata.get('title', 'N/A')} "
                          f"({c.metadata['chunk_size']} chars)")
 
            context_parts = []
            for c in retrieved:
                label = c.metadata.get('title') or c.metadata.get('source_file', '')
                context_parts.append(f'[{label}]\n{c.content}')
            context = '\n\n'.join(context_parts)
 
            # Build RAG-grounded system prompt
            if original_lang == 'ar':
                system_msg = f"""{mode_block}
 
أنت مساعد متخصص في منتجات وخدمات التأمين لشركة التعاونية للتأمين.
مهمتك استخراج جميع المعلومات ذات الصلة من النصوص المقدمة فقط.
لا تضف أي معلومات من خارج النص.
 
يجب عليك:
- الإجابة باللهجة السعودية العامية دائماً (مثل: وش، كيف، إيش، عشان، بس، زين، الحين، إن شاء الله، ما عندي، تقدر، ودّك).
- استخراج جميع العناصر المذكورة في النص دون حذف أي منها.
- عدم تكرار أي بند.
- عرض الإجابة في نقاط واضحة إذا كانت موجودة في النص."""
 
                user_prompt = f"""النصوص المرجعية:
 
{context}
 
السؤال:
{query}
 
أجب الآن باللهجة السعودية."""
 
            else:
                system_msg = f"""{mode_block}
 
You are a professional insurance assistant specializing in Tawuniya Insurance products.
Your task is to extract ALL relevant information strictly from the provided texts and answer in English.
Do NOT add information from outside the texts.
 
CRITICAL LANGUAGE RULE:
- YOUR ENTIRE ANSWER MUST BE IN ENGLISH. Do not include any Arabic text in your response.
- Translate all relevant content from Arabic to English completely.
 
If the question requires multiple items, list ALL of them without omission or repetition."""
 
                user_prompt = f"""Source texts:
 
{context}
 
Question (answer in English only):
{query}
 
Provide your answer now in English."""
 
        else:
            # ── Conversational path: no retrieval, mode instructions only ──
            if debug:
                print(f'[DEBUG] Path: Conversational — no retrieval, mode instructions only')
 
            if original_lang == 'ar':
                system_msg = f"""{mode_block}
 
أنت مساعد متخصص في منتجات وخدمات التأمين لشركة التعاونية للتأمين.
الإجابة باللهجة السعودية العامية دائماً (مثل: وش، كيف، إيش، عشان، بس، زين، الحين، إن شاء الله، ما عندي، تقدر، ودّك).
تعليمات حاسمة: أجب حصراً وفق أسلوب الرد والأمثلة الواردة أعلاه.
لا تستخدم أي معلومات من معرفتك العامة أو من خارج التعليمات أعلاه.
إذا احتاج الرد إلى تفاصيل واقعية غير موجودة في التعليمات، اطرح سؤالاً توضيحياً على المستخدم بدلاً من اختلاق معلومات."""
 
                user_prompt = f"""{query}"""
 
            else:
                system_msg = f"""{mode_block}
 
You are a professional insurance assistant specializing in Tawuniya Insurance products.
CRITICAL: Reply STRICTLY following the mode instructions and examples above.
Do NOT use any knowledge from your training data. Do NOT invent facts, policies, or product details.
If specific factual detail is needed that is not in the mode instructions, ask the user a clarifying question instead.
YOUR ENTIRE ANSWER MUST BE IN ENGLISH."""
 
                user_prompt = f"""{query}"""
 
        # ── Step 5: generate answer ────────────────────────────────────
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user',   'content': user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        answer = response.choices[0].message.content
 
        # ── Step 6: update history ─────────────────────────────────────
        self.conversation_history.append({'role': 'user',      'content': standalone_query})
        self.conversation_history.append({'role': 'assistant', 'content': answer})
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
 
        return (answer, context) if return_context else answer
 
    # ------------------------------------------------------------------
    # Follow-up question generation
    # ------------------------------------------------------------------
 
    def generate_followup_questions(
        self,
        query: str,
        answer: str,
        context: str,
        lang: str,
    ) -> List[str]:
        """Generate 3 context-bounded follow-up questions."""
        lang_name = 'Arabic' if lang == 'ar' else 'English'
        system_prompt = (
            f'You are an assistant helping users navigate Tawuniya Insurance information. '
            f'Based on the user\'s question, the answer, and ONLY the following texts, '
            f'suggest exactly 3 short follow-up questions.\n'
            f'CRITICAL: questions must be answerable from the provided texts only.\n'
            f'Output JSON: {{"questions": ["Q1?", "Q2?", "Q3?"]}}\n'
            f'Questions MUST be in {lang_name}.\n\n'
            f'Texts:\n{context}'
        )
        user_prompt = (
            f'User Question: {query}\nChatbot Answer: {answer}\n\n'
            f'Provide the 3 follow-up questions as JSON.'
        )
        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_prompt},
                ],
                temperature=0.3,
                max_tokens=200,
                response_format={'type': 'json_object'},
            )
            result = json.loads(response.choices[0].message.content.strip())
            return result.get('questions', [])
        except Exception as e:
            print(f'Follow-up generation failed: {e}')
            return []
 
    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
 
    def get_statistics(self) -> Dict:
        sizes = [c.metadata['chunk_size'] for c in self.chunks]
        return {
            'total_chunks':   len(self.chunks),
            'avg_chunk_size': round(sum(sizes) / len(sizes), 1) if sizes else 0,
            'max_chunk_size': max(sizes) if sizes else 0,
            'min_chunk_size': min(sizes) if sizes else 0,
        }
 
 
# ========== ENTRY POINT ==========
 
if __name__ == '__main__':
    CHUNKS_FOLDER = os.path.join('splitted_chunks', 'splitted_chunks')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
 
    engine = RAGEngine(
        chunks_folder=CHUNKS_FOLDER,
        openai_api_key=OPENAI_API_KEY,
    )
 
    stats = engine.get_statistics()
    print('\n' + '=' * 70)
    print('STATISTICS')
    print('=' * 70)
    for k, v in stats.items():
        print(f'  {k}: {v}')
 
    print('\n' + '=' * 70)
    print('EXAMPLE QUERIES')
    print('=' * 70)
 
    print('\n1. Arabic query:')
    answer = engine.answer_question('ما هي مزايا برنامج تأمين العمرة؟', debug=True)
    print(f'\nAnswer:\n{answer}')
 
    print('\n\n2. English query:')
    answer = engine.answer_question('What does medical malpractice insurance cover?', debug=True)
    print(f'\nAnswer:\n{answer}')