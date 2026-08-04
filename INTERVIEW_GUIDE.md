# Interview Guide: Explaining This Project

## The 5-Minute Explanation (say this out loud, practice it)

> "I built an AI Personal Study & Interview Assistant — a Python app that helps someone learn any technical topic and prepare for interviews on it. You type in a topic like 'SQL Joins' or 'React Hooks,' and it generates beginner-through-advanced explanations, realistic interview questions with model answers, an auto-graded multiple-choice quiz, flashcards, real-world examples, and coding exercises — all through the OpenAI API orchestrated with LangChain.
>
> The second half is a RAG feature: you can upload your own PDF — lecture notes, a textbook chapter, whatever — and ask it questions. It extracts the text, chunks it, embeds the chunks locally with Sentence-Transformers, indexes them in FAISS, and at query time retrieves the most relevant chunks and asks the LLM to answer *grounded only in that content*, citing page numbers. It keeps a rolling conversation memory so follow-up questions work naturally.
>
> Everything is persisted in SQLite — chat history and every generated note, scoped per session — and you can export any topic's notes as a downloadable PDF.
>
> Architecturally, I built it in layers: a thin Streamlit UI, an agent layer that owns the prompting logic for each content type, a service layer that wraps the actual OpenAI, FAISS, and PDF I/O calls, and a repository-pattern SQLite layer. Every agent takes its dependencies through the constructor rather than creating them itself, so I could unit-test the agents with a mocked LLM — I've got 45 tests that run fully offline in about two seconds.
>
> A few production details I'm proud of: config is validated at startup and fails fast with a clear message if the API key is missing, rather than failing three calls deep. LLM calls retry with exponential backoff on rate limits and connection errors specifically — not on auth errors, since retrying a bad key just wastes time. And LLM JSON output is parsed defensively and validated through Pydantic schemas, since you can't fully trust an LLM to always return well-formed JSON even when you ask it to."

---

## Common Interview Questions & Answers

**Q: Why LangChain instead of calling the OpenAI SDK directly?**
A: LangChain gives a standard message abstraction (`SystemMessage`/`HumanMessage`/`AIMessage`) that isn't tied to one provider's request shape, so swapping providers later means changing one service file, not every prompt call site. In this project it's a thin layer — I could justify either choice — but it demonstrates the pattern.

**Q: Why FAISS instead of a hosted vector database?**
A: No external service dependency, no cost, and it's trivial for anyone to clone the repo and run it end-to-end. FAISS with `IndexFlatIP` does exact nearest-neighbor search, which is completely fine at the scale of a few thousand PDF chunks — approximate indexes like HNSW only start mattering at millions of vectors.

**Q: How do you keep the vector store and the actual text in sync?**
A: FAISS only stores vectors, not the original text, so I keep a parallel Python list of texts/metadata and persist both together — the FAISS index as a `.faiss` file and the sidecar list as a pickle — under the same base filename. That's the standard pattern for using raw FAISS in a RAG pipeline (as opposed to a wrapper like Chroma that handles this for you).

**Q: How do you handle the LLM returning malformed JSON for quizzes/flashcards?**
A: `LLMService.generate_json()` explicitly instructs the model to return only JSON, strips markdown code fences defensively if the model adds them anyway, and if `json.loads()` still fails, raises a clear `LLMServiceError` that the UI surfaces as an error message rather than crashing. On top of that, everything parsed is validated through Pydantic models (e.g. `QuizQuestion` checks that `correct_answer_index` actually points at a real option), so even syntactically valid but semantically broken JSON gets caught.

**Q: How is retry logic implemented, and why not retry everything?**
A: I used `tenacity` with exponential backoff, scoped only to `RateLimitError` and `APIConnectionError` — transient failures. `AuthenticationError` is explicitly excluded, because retrying a bad API key four times with backoff just delays the inevitable failure by 20+ seconds for no benefit. Each exception type maps to a specific, user-readable error message.

**Q: Why SQLite instead of Postgres?**
A: For a single-user, locally-run resume project, SQLite has zero setup cost and is more than sufficient. I used the repository pattern in `sqlite_db.py` specifically so swapping to Postgres later is a matter of changing the connection layer, not touching any agent or UI code — I call that out as a named future improvement.

**Q: Why open a new SQLite connection per operation instead of one long-lived connection?**
A: SQLite connections aren't safe to share across threads, and Streamlit can invoke callbacks on different threads. A context-manager-scoped connection per operation avoids a subtle concurrency bug at the cost of a small amount of overhead, which is the right tradeoff at this scale.

**Q: Why is `Settings` a frozen dataclass instead of just module-level variables or a dict?**
A: Immutability prevents any part of the app from accidentally mutating shared config at runtime, which is a real source of hard-to-trace bugs in larger codebases. It also gives IDE autocomplete and type-checking that a dict wouldn't. I load and validate it exactly once in a module-level singleton, so every file gets the same validated config for free.

**Q: What would you change if this needed to support many concurrent users?**
A: Move off SQLite to Postgres, add real authentication instead of a random session UUID, move the FAISS index per-user or add proper multi-tenancy to the vector store, and put the LLM calls behind a queue/worker so one user's slow generation doesn't block others — Streamlit's single-process model doesn't scale well past a handful of concurrent users, so I'd likely put a FastAPI backend in front and make Streamlit (or a proper frontend) just a client of it.

**Q: How did you test this without spending money on the OpenAI API?**
A: Every test mocks the `ChatOpenAI` client directly (`unittest.mock.MagicMock`), so `LLMService`, and every agent that depends on it, is tested with dependency injection — I pass a mock in place of a real `LLMService`. The full 45-test suite runs in about two seconds with zero network calls and zero cost.

**Q: Walk me through what happens end-to-end when I upload a PDF and ask a question.**
A: (See `ARCHITECTURE.md` → "Data Flow: Upload PDF & Ask a Question" for the full numbered breakdown — extract → chunk → embed → index → retrieve → ground → answer → persist.)
