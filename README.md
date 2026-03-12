# Tawuniya Insurance Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for Tawuniya Insurance — answers questions in Arabic and English using your own scraped content.

---

## How it works

1. Scrape Tawuniya website → raw text
2. Clean the text (`cleaner.py`)
3. Split into chunks (`splitted_chunks/`)
4. At runtime, the API loads chunks, builds embeddings (FAISS) + BM25 index, then answers questions using GPT-4o-mini

---

## Requirements

- Python 3.10+
- OpenAI API key

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd tawuniya_chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## Run

```bash
uvicorn api:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | Ask a question `{"query": "..."}` |
| POST | `/clear` | Clear conversation history |
| GET | `/stats` | Show chunk statistics |

---

## Data Pipeline (one-time setup)

```bash
# Clean raw scraped text
python cleaner.py

# The splitted_chunks/ folder should already contain .txt chunk files
```

---

## Project Structure

```
api.py              # FastAPI server
engine.py           # RAG engine (FAISS + BM25 + GPT-4o-mini)
cleaner.py          # Text cleaning utility
static/index.html   # Chat UI
splitted_chunks/    # Pre-split knowledge base chunks
```
