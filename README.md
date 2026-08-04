# 🎓 AI Personal Study & Interview Assistant

A production-quality, agent-based AI application that helps you learn any technical topic and prepare for interviews — with multi-level explanations, auto-generated interview Q&A, quizzes, flashcards, coding exercises, and **RAG-based Q&A over your own PDF study materials**.

Built with **Python, LangChain, OpenAI, FAISS, Sentence Transformers, and Streamlit**, following clean architecture and OOP principles.

---

## ✨ Features

- 🧠 Enter any topic → get **beginner / intermediate / advanced** explanations
- 💼 Auto-generated **interview questions & model answers**
- 📝 Auto-generated **multiple-choice quizzes** with instant grading
- 🃏 **Flashcards** for spaced-repetition style review
- 🌍 **Real-world examples** of how the topic is used in industry
- 💻 **Coding exercises** with starter code, solutions, and hints
- 📄 **Upload a PDF** and ask questions about it (Retrieval-Augmented Generation)
- 💬 **Conversation memory** for natural follow-up questions
- 🗄️ **SQLite-backed** chat history and saved notes, per session
- ⬇️ **Download generated notes as a PDF**
- 🌙 **Dark mode UI** with loading indicators throughout
- 🛡️ Graceful error handling for API failures, bad uploads, and malformed AI output
- ✅ Unit-tested core logic (45+ tests, LLM calls mocked — runs offline)

---

## 🏗️ Architecture

This project follows a **layered / clean architecture**:

```
UI Layer        →  app.py (Streamlit)
Agent Layer      →  agents/  (business logic: what to ask the LLM, how to structure it)
Service Layer     →  services/ (LLM calls, embeddings, vector search, PDF I/O)
Data Layer         →  database/ (SQLite repository), vector_db/ (FAISS index)
Cross-cutting      →  config.py, logger.py, models/schemas.py, utils/
```

Each layer only depends on the layer below it. Agents never import `openai` or `sqlite3` directly — they depend on `LLMService` and `SQLiteDatabase` through constructor injection, which makes every agent independently unit-testable with mocks.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for full diagrams.

---

## 📁 Project Structure

```
AI-Study-Agent/
├── app.py                     # Streamlit UI — composition root
├── config.py                  # Centralized, validated settings (singleton)
├── logger.py                  # Centralized logging (console + rotating file)
├── requirements.txt
├── .env.example
├── README.md
│
├── agents/                    # Business logic — what to generate & how
│   ├── study_agent.py          # Explanations, examples, coding exercises
│   ├── interview_agent.py      # Interview Q&A
│   ├── quiz_agent.py           # Quizzes & flashcards
│   └── pdf_agent.py            # RAG Q&A over uploaded PDFs + memory
│
├── services/                  # External integrations
│   ├── llm_service.py          # OpenAI/LangChain wrapper, retries, JSON mode
│   ├── embedding_service.py    # Sentence Transformers wrapper
│   ├── vector_service.py       # FAISS index management
│   └── pdf_service.py          # PDF reading & chunking
│
├── database/
│   └── sqlite_db.py            # Repository pattern for chat history & notes
│
├── models/
│   └── schemas.py               # Pydantic models for LLM output validation
│
├── utils/
│   ├── validators.py            # Input validation
│   ├── pdf_export.py            # Notes → downloadable PDF
│   └── session.py               # Session ID helpers
│
├── tests/                      # 45+ unit tests, offline (mocked LLM calls)
├── data/, uploads/, notes/, vector_db/, database/, assets/
```

---

## 🚀 Installation

### 1. Prerequisites
- Python 3.12+
- An OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 2. Clone & set up a virtual environment
```bash
git clone <your-repo-url> AI-Study-Agent
cd AI-Study-Agent
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Open `.env` and set your key:
```
GEMINI_API_KEY=sk-...your-real-key...
```

### 5. Run the app
```bash
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

---

## 🔑 How to Get an OpenAI API Key

1. Go to https://platform.openai.com/signup and create an account (or log in).
2. Navigate to **https://platform.openai.com/api-keys**.
3. Click **"Create new secret key"**, name it, and copy it immediately (it's shown only once).
4. Add billing info under **Settings → Billing** — the API is pay-as-you-go (a few cents will cover extensive testing with `gpt-4o-mini`).
5. Paste the key into your `.env` file as `GEMINI_API_KEY`.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```
All LLM calls are mocked, so tests run fully offline in ~2 seconds with zero API cost.

---

## 🖼️ What to Expect (UI Walkthrough)

- **Study Generator tab**: enter a topic (e.g. "Python Decorators"), pick which artifacts to generate, click Generate. Content streams in section-by-section with a spinner/status indicator per artifact type, then renders as tabs (levels), expanders (Q&A, quiz, flashcards), and an interactive quiz with instant right/wrong feedback.
- **PDF Q&A tab**: drag-and-drop a PDF, click "Index this PDF," then chat with it using a standard chat interface. Answers cite the page number they were grounded in.
- **History tab**: browse past conversations and every previously generated note, grouped by topic.
- Dark theme throughout; a "Download as PDF" button on every generated topic.

---

## 📊 Sample Input/Output

**Input (Study Generator):** Topic = `"SQL Joins"`, types = Explanations + Interview Q&A + Quiz

**Output (abridged):**
> **Beginner:** "A SQL JOIN combines rows from two or more tables based on a related column between them. Think of it like matching entries in two spreadsheets..."
>
> **Interview Q (advanced):** "What's the performance difference between a LEFT JOIN and a subquery-based approach for a filtered join, and when would you prefer one over the other?"
>
> **Quiz Q1:** "Which JOIN type returns all rows from the left table, and matched rows from the right table, with NULLs where there's no match?" → Options: [INNER JOIN, LEFT JOIN, RIGHT JOIN, CROSS JOIN] → Correct: LEFT JOIN

**Input (PDF Q&A):** Upload `machine_learning_notes.pdf`, ask *"What does the document say about overfitting?"*
**Output:** A grounded answer citing the specific page(s) the model retrieved, plus a low-confidence warning if the retrieved chunks weren't strongly related to the question.

---

## 🔮 Future Improvements

- Swap SQLite for PostgreSQL + add user authentication for true multi-user support
- Stream LLM tokens to the UI in real time instead of waiting for full completion
- Add spaced-repetition scheduling (SM-2 algorithm) for flashcards
- Support OCR for scanned/image-only PDFs
- Add a FastAPI backend so the same agents can power a mobile app, not just Streamlit
- Cache generated content by topic hash to cut redundant API costs
- Add Anthropic/local-model provider option behind the existing `LLMService` interface

---

## 🧑‍💻 Author's Note

This project was built to demonstrate: clean layered architecture, dependency injection for testability, production-grade error handling (retry logic, graceful degradation, user-safe error messages), Retrieval-Augmented Generation with a local-embedding + FAISS pipeline, and structured LLM output validation via Pydantic.
