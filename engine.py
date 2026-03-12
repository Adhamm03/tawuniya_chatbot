import re
import os
import glob
import numpy as np
import faiss
from dataclasses import dataclass
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI
import json


@dataclass
class Chunk:
    content: str
    metadata: Dict


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

    def answer_question(
        self,
        query: str,
        debug: bool = False,
        return_context: bool = False,
    ):
        """Answer a question using RAG with conversation memory."""
        original_lang = self.detect_language(query)

        standalone_query = self.rewrite_query_with_history(query)
        if debug and standalone_query != query:
            print(f'[DEBUG] Original:  {query}')
            print(f'[DEBUG] Rewritten: {standalone_query}')

        if original_lang == 'en':
            retrieval_query = self.translate_to_arabic(standalone_query)
            if debug:
                print(f'[DEBUG] Translated for retrieval: {retrieval_query}')
        else:
            retrieval_query = standalone_query

        retrieved = self.retrieve_chunks(retrieval_query, top_k=10)

        if not retrieved:
            no_result = (
                'عذراً، لم أتمكن من العثور على معلومات ذات صلة بسؤالك.'
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

        if original_lang == 'ar':
            system_msg = """أنت مساعد متخصص في منتجات وخدمات التأمين لشركة التعاونية للتأمين.
مهمتك استخراج جميع المعلومات ذات الصلة من النصوص المقدمة فقط.
لا تضف أي معلومات من خارج النص.

يجب عليك:
- الإجابة بنفس لغة السؤال تماماً (عربي فقط إذا كان السؤال بالعربية).
- استخراج جميع العناصر المذكورة في النص دون حذف أي منها.
- عدم تكرار أي بند.
- عرض الإجابة في نقاط واضحة إذا كانت موجودة في النص."""

            user_prompt = f"""النصوص المرجعية:

{context}

السؤال:
{query}

أجب الآن."""

        else:
            system_msg = """You are a professional insurance assistant specializing in Tawuniya Insurance products.
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
