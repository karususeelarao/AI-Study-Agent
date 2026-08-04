# Architecture & Data Flow

## System Architecture (ASCII)

```
                                   ┌────────────────────────────┐
                                   │      Streamlit UI (app.py)  │
                                   │  Study | PDF Q&A | History   │
                                   └───────────────┬────────────┘
                                                    │
                     ┌──────────────────────────────┼──────────────────────────────┐
                     │                              │                              │
             ┌───────▼────────┐            ┌────────▼────────┐            ┌────────▼────────┐
             │  StudyAgent     │            │  InterviewAgent  │            │   QuizAgent      │
             │  explanations,  │            │  interview Q&A   │            │  quiz, flashcards │
             │  examples, code │            │                  │            │                  │
             └───────┬────────┘            └────────┬────────┘            └────────┬────────┘
                     │                              │                              │
                     └──────────────┬───────────────┴───────────────┬──────────────┘
                                     │                               │
                             ┌───────▼────────┐              ┌──────▼───────┐
                             │   LLMService     │              │  PDFAgent     │
                             │  (OpenAI/LangChain,│            │  RAG + memory  │
                             │   retries, JSON)  │              └──────┬───────┘
                             └───────┬────────┘                       │
                                     │                     ┌──────────┼───────────┐
                                     │                     │                      │
                                     │             ┌───────▼──────┐      ┌────────▼───────┐
                                     │             │  PDFService    │      │ VectorService   │
                                     │             │  read+chunk    │      │  FAISS index     │
                                     │             └────────────────┘      └────────┬────────┘
                                     │                                              │
                                     │                                     ┌────────▼────────┐
                                     │                                     │ EmbeddingService  │
                                     │                                     │ Sentence-Transf.   │
                                     │                                     └───────────────────┘
                                     │
                             ┌───────▼────────┐
                             │   OpenAI API      │
                             └───────────────────┘

             ┌──────────────────────────────────────────────────────────┐
             │                     SQLiteDatabase                        │
             │        chat_messages table  |  study_notes table           │
             └──────────────────────────────────────────────────────────┘
                     ▲ every agent interaction is persisted here ▲

             Cross-cutting (used by every layer above):
             config.py (Settings singleton) · logger.py (rotating file + console)
             models/schemas.py (Pydantic validation) · utils/ (validators, pdf export, session)
```

**Layering rule:** UI → Agents → Services → External APIs / Data stores. Nothing skips a layer (e.g., `app.py` never calls `openai` directly — always through an Agent, which goes through `LLMService`).

---

## Data Flow: "Generate Study Material" (Text Generation Path)

```
1. User types topic in Streamlit → validators.validate_topic()
2. app.py calls StudyAgent.explain_all_levels(topic)
3. StudyAgent builds a level-specific prompt → LLMService.generate()
4. LLMService converts prompt to LangChain messages → ChatOpenAI.invoke()
     - tenacity retries on RateLimitError / APIConnectionError
     - AuthenticationError raised immediately as LLMServiceError (no retry)
5. OpenAI API returns completion text
6. StudyAgent wraps it in a validated TopicExplanation (Pydantic-style dataclass)
7. app.py renders it in a Streamlit tab
8. SQLiteDatabase.save_note() persists it, scoped to session_id
```

## Data Flow: "Upload PDF & Ask a Question" (RAG Path)

```
1. User uploads PDF → validators.validate_pdf_filename() + size check
2. PDFService.save_upload() writes bytes to /uploads
3. PDFAgent.ingest_pdf():
     a. PDFService.extract_text_by_page()   → raw text per page
     b. RecursiveCharacterTextSplitter       → overlapping ~1000-char chunks
     c. EmbeddingService.embed_texts()       → local Sentence-Transformers vectors
     d. VectorService.add_texts()            → vectors added to FAISS IndexFlatIP
     e. VectorService.save()                 → index + sidecar text persisted to /vector_db
4. User asks a question in chat_input
5. PDFAgent.ask():
     a. EmbeddingService.embed_query()       → embed the question
     b. VectorService.similarity_search()    → top-k nearest chunks (cosine via inner product)
     c. Build a context-grounded prompt      → "answer ONLY using this context"
     d. LLMService.generate(..., history=self._memory)  → grounded answer
     e. Rolling memory window updated (last 6 turns) for follow-up questions
6. Answer + confidence note rendered in chat; both turns saved to SQLite
```

---

## Why FAISS + Sentence-Transformers instead of a hosted vector DB / OpenAI embeddings

| Choice | Alternative | Why this project uses it |
|---|---|---|
| FAISS (local, file-based) | Pinecone / Weaviate (hosted) | Zero external dependency, zero cost, runs entirely offline once the model is downloaded — ideal for a resume project reviewers can run themselves |
| Sentence-Transformers (local embeddings) | OpenAI `text-embedding-3-*` | Free, no network round-trip per chunk, keeps document content off third-party APIs if desired |
| `IndexFlatIP` (exact search) | `IndexIVFFlat` / HNSW (approximate) | Exact search is fine at this project's scale (hundreds–thousands of chunks); approximate indexes only pay off at millions of vectors |
