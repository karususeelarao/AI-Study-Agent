"""
app.py
==============================================================================
AI Personal Study & Interview Assistant — Streamlit Frontend

This is the composition root: it wires together config, services, agents,
and the database, and renders the UI. Business logic lives in
agents/services; this file is intentionally "thin" — it only handles UI
state and delegates actual work.
==============================================================================
"""

from __future__ import annotations

import streamlit as st
from ui.dashboard import render_dashboard
from ui.sidebar import render_sidebar
from ui.theme import load_theme

from config import settings
from database.sqlite_db import SQLiteDatabase
from logger import get_logger
from models.schemas import DifficultyLevel
from agents.interview_agent import InterviewAgent, InterviewAgentError
from agents.interview_mock_agent import MockInterviewAgent, MockInterviewAgentError
from agents.pdf_agent import PDFAgent, PDFAgentError
from agents.quiz_agent import QuizAgent, QuizAgentError
from agents.study_agent import StudyAgent, StudyAgentError
from services.embedding_service import EmbeddingService, EmbeddingServiceError
from services.llm_service import LLMService, LLMServiceError
from services.pdf_service import PDFService, PDFServiceError
from services.vector_service import VectorService, VectorServiceError
from utils.pdf_export import PDFExportError, export_notes_to_pdf
from utils.session import new_session_id
from utils.validators import ValidationError, validate_pdf_filename, validate_question, validate_topic

logger = get_logger(__name__)

st.set_page_config(
    page_title=settings.app_name,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Dark mode styling
# ==============================================================================
def inject_dark_theme() -> None:
    """Inject custom CSS for a polished dark-mode UI."""
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        section[data-testid="stSidebar"] {
            background-color: #161a23;
        }
        .stButton>button {
            background-color: #6c5ce7;
            color: white;
            border-radius: 8px;
            border: none;
            font-weight: 600;
            padding: 0.5rem 1.2rem;
        }
        .stButton>button:hover {
            background-color: #5849c4;
            color: white;
        }
        .study-card {
            background-color: #1b1f2a;
            border: 1px solid #2a2f3d;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }
        .flashcard-front {
            background-color: #23283a;
            border-radius: 10px;
            padding: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .flashcard-back {
            background-color: #17321f;
            border-radius: 10px;
            padding: 1.2rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )




# ==============================================================================
# Cached resource construction (loaded once per process, not per re-run)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_database() -> SQLiteDatabase:
    return SQLiteDatabase()


@st.cache_resource(show_spinner=False)
def get_llm_service() -> LLMService:
    return LLMService()


@st.cache_resource(show_spinner=False)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@st.cache_resource(show_spinner=False)
def get_pdf_service() -> PDFService:
    return PDFService()


@st.cache_resource(show_spinner=False)
def get_vector_service(_embedding_service: EmbeddingService) -> VectorService:
    vector_service = VectorService(embedding_service=_embedding_service)
    vector_service.load()  # restore previously indexed PDFs, if any
    return vector_service


def get_agents():
    """Construct agents fresh each run (cheap: they just hold references to cached services)."""
    llm = get_llm_service()
    embedder = get_embedding_service()
    pdf_service = get_pdf_service()
    vector_service = get_vector_service(embedder)

    study_agent = StudyAgent(llm)
    interview_agent = InterviewAgent(llm)
    quiz_agent = QuizAgent(llm)
    mock_interview_agent = MockInterviewAgent(llm)

    if "pdf_agent" not in st.session_state:
        st.session_state.pdf_agent = PDFAgent(llm, pdf_service, vector_service)

    return (
    study_agent,
    interview_agent,
    quiz_agent,
    mock_interview_agent,
    st.session_state.pdf_agent,
    pdf_service,
)


# ==============================================================================
# Session state initialization
# ==============================================================================
def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
        logger.info("New session started: %s", st.session_state.session_id)
    if "generated" not in st.session_state:
        st.session_state.generated = {}  # topic -> generated content dict
    if "pdf_indexed" not in st.session_state:
        st.session_state.pdf_indexed = False
    if "pdf_chat" not in st.session_state:
        st.session_state.pdf_chat = []


# ==============================================================================
# Study material generation page
# ==============================================================================
def render_study_tab(study_agent, interview_agent, quiz_agent, db: SQLiteDatabase) -> None:
    st.subheader("📘 Generate Study Material")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input(
            "Enter a topic", placeholder="e.g. Python Decorators, SQL Joins, React Hooks"
        )
    with col2:
        content_types = st.multiselect(
            "Generate",
            options=["Explanations", "Interview Q&A", "Quiz", "Flashcards", "Examples", "Coding Exercises"],
            default=["Explanations", "Interview Q&A"],
        )

    generate_clicked = st.button("🚀 Generate", type="primary")

    if generate_clicked:
        try:
            topic = validate_topic(topic_input)
            db.add_chat_message(
                st.session_state.session_id,
                "user",
                f"Generate study material for: {topic}"
)
        except ValidationError as exc:
            st.error(f"⚠️ {exc}")
            return

        result: dict = {"topic": topic}
        try:
            with st.spinner(f"🧠 Generating study material for '{topic}'..."):
                if "Explanations" in content_types:
                    with st.status("Writing explanations at all 3 levels...", expanded=False):
                        result["explanations"] = study_agent.explain_all_levels(topic)

                if "Interview Q&A" in content_types:
                    with st.status("Drafting interview questions...", expanded=False):
                        result["interview_qa"] = interview_agent.generate_all_levels(topic, count_per_level=3)

                if "Quiz" in content_types:
                    with st.status("Building quiz...", expanded=False):
                        result["quiz"] = quiz_agent.generate_quiz(topic, DifficultyLevel.INTERMEDIATE, count=5)

                if "Flashcards" in content_types:
                    with st.status("Making flashcards...", expanded=False):
                        result["flashcards"] = quiz_agent.generate_flashcards(topic, count=8)

                if "Examples" in content_types:
                    with st.status("Finding real-world examples...", expanded=False):
                        result["examples"] = study_agent.generate_real_world_examples(topic)

                if "Coding Exercises" in content_types:
                    with st.status("Writing coding exercises...", expanded=False):
                        result["exercises"] = study_agent.generate_coding_exercises(
                            topic, DifficultyLevel.INTERMEDIATE
                        )

            st.session_state.generated[topic] = result
            _persist_generated_content(db, topic, result)
            db.add_chat_message(
                st.session_state.session_id,
                "assistant",
                f"Generated study material for '{topic}'."
)
            st.success(f"✅ Study material for '{topic}' generated and saved!")

        except (StudyAgentError, InterviewAgentError, QuizAgentError, LLMServiceError) as exc:
            logger.error("Generation failed: %s", exc)
            st.error(f"❌ Generation failed: {exc}")
            return

    # Render whichever topic was last generated (or let user pick from history)
    saved_topics = db.get_all_topics(st.session_state.session_id)
    if saved_topics:
        st.divider()
        pick = st.selectbox("📂 View previously generated topics", ["(current)"] + saved_topics)
        if pick != "(current)" and pick not in st.session_state.generated:
            st.session_state.generated[pick] = _load_generated_from_db(db, pick)

    for topic, content in st.session_state.generated.items():
        _render_generated_content(topic, content, db)


def _persist_generated_content(db: SQLiteDatabase, topic: str, result: dict) -> None:
    """Save every generated artifact for this topic into SQLite."""
    session_id = st.session_state.session_id
    for exp in result.get("explanations", []):
        db.save_note(session_id, topic, "explanation", exp.level.value, exp.content)
    for qa in result.get("interview_qa", []):
        db.save_note(
            session_id, topic, "interview_qa", qa.difficulty.value, f"Q: {qa.question}\nA: {qa.answer}"
        )
    for i, q in enumerate(result.get("quiz", [])):
        content = f"Q: {q.question}\nOptions: {q.options}\nCorrect: {q.options[q.correct_answer_index]}\nWhy: {q.explanation}"
        db.save_note(session_id, topic, "quiz", "intermediate", content)
    for card in result.get("flashcards", []):
        db.save_note(session_id, topic, "flashcard", "n/a", f"Front: {card.front}\nBack: {card.back}")
    for example in result.get("examples", []):
        db.save_note(session_id, topic, "example", "n/a", example)
    for ex in result.get("exercises", []):
        db.save_note(
            session_id, topic, "exercise", "intermediate",
            f"{ex.title}\n{ex.problem_statement}\n\nSolution:\n{ex.solution}",
        )


def _load_generated_from_db(db: SQLiteDatabase, topic: str) -> dict:
    """Reconstruct a lightweight display dict from raw saved notes (read-only view)."""
    notes = db.get_notes(st.session_state.session_id, topic=topic)
    grouped: dict = {"topic": topic, "raw_notes": notes}
    return grouped


def _render_generated_content(topic: str, content: dict, db: SQLiteDatabase) -> None:
    st.divider()
    st.markdown(f"## 📖 {topic}")

    if "raw_notes" in content:
        for note in content["raw_notes"]:
            st.markdown(
                f"<div class='study-card'><b>{note.note_type} ({note.level})</b><br>{note.content}</div>",
                unsafe_allow_html=True,
            )
        return

    if content.get("explanations"):
        tabs = st.tabs([e.level.value.capitalize() for e in content["explanations"]])
        for tab, exp in zip(tabs, content["explanations"]):
            with tab:
                st.markdown(exp.content)

    if content.get("interview_qa"):
        with st.expander("💼 Interview Questions & Answers", expanded=False):
            for qa in content["interview_qa"]:
                st.markdown(f"**Q ({qa.difficulty.value}):** {qa.question}")
                st.markdown(f"**A:** {qa.answer}")
                st.markdown("---")

    if content.get("quiz"):
        with st.expander("📝 Quiz", expanded=False):
            for i, q in enumerate(content["quiz"], start=1):
                st.markdown(f"**{i}. {q.question}**")
                choice = st.radio(
                    "Choose one:", q.options, key=f"{topic}_quiz_{i}", label_visibility="collapsed"
                )
                if st.button("Check answer", key=f"{topic}_check_{i}"):
                    correct = q.options[q.correct_answer_index]
                    if choice == correct:
                        st.success(f"✅ Correct! {q.explanation}")
                    else:
                        st.error(f"❌ Incorrect. Correct answer: {correct}. {q.explanation}")

    if content.get("flashcards"):
        with st.expander("🃏 Flashcards", expanded=False):
            for i, card in enumerate(content["flashcards"], start=1):
                st.markdown(f"<div class='flashcard-front'>Q{i}: {card.front}</div>", unsafe_allow_html=True)
                with st.expander("Show answer", expanded=False):
                    st.markdown(f"<div class='flashcard-back'>{card.back}</div>", unsafe_allow_html=True)

    if content.get("examples"):
        with st.expander("🌍 Real-World Examples", expanded=False):
            for ex in content["examples"]:
                st.markdown(f"- {ex}")

    if content.get("exercises"):
        with st.expander("💻 Coding Exercises", expanded=False):
            for ex in content["exercises"]:
                st.markdown(f"**{ex.title}**")
                st.markdown(ex.problem_statement)
                if ex.starter_code:
                    st.code(ex.starter_code)
                with st.expander("Show solution"):
                    st.code(ex.solution)
                if ex.hints:
                    st.caption("Hints: " + " | ".join(ex.hints))

    # Download as PDF
    if st.button(f"⬇️ Download '{topic}' notes as PDF", key=f"download_{topic}"):
        try:
            sections = _build_pdf_sections(content)
            pdf_path = export_notes_to_pdf(topic, sections)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Click to save PDF",
                    data=f.read(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"dl_btn_{topic}",
                )
        except PDFExportError as exc:
            st.error(f"❌ Could not export PDF: {exc}")


def _build_pdf_sections(content: dict) -> dict[str, str]:
    """Flatten generated content into {heading: text} for PDF export."""
    sections: dict[str, str] = {}
    for exp in content.get("explanations", []):
        sections[f"Explanation ({exp.level.value})"] = exp.content
    if content.get("interview_qa"):
        sections["Interview Q&A"] = "\n\n".join(
            f"Q: {qa.question}\nA: {qa.answer}" for qa in content["interview_qa"]
        )
    if content.get("quiz"):
        sections["Quiz"] = "\n\n".join(
            f"{q.question}\nOptions: {', '.join(q.options)}\n"
            f"Correct: {q.options[q.correct_answer_index]}\n{q.explanation}"
            for q in content["quiz"]
        )
    if content.get("flashcards"):
        sections["Flashcards"] = "\n\n".join(f"{c.front} -> {c.back}" for c in content["flashcards"])
    if content.get("examples"):
        sections["Real-World Examples"] = "\n".join(f"- {e}" for e in content["examples"])
    if content.get("exercises"):
        sections["Coding Exercises"] = "\n\n".join(
            f"{e.title}\n{e.problem_statement}\n\nSolution:\n{e.solution}" for e in content["exercises"]
        )
    return sections


# ==============================================================================
# PDF Q&A (RAG) page
# ==============================================================================
def render_pdf_tab(pdf_agent: PDFAgent, pdf_service: PDFService, db: SQLiteDatabase) -> None:
    st.subheader("📄 Upload a PDF & Ask Questions")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file is not None:
        try:
            validate_pdf_filename(uploaded_file.name)
            file_bytes = uploaded_file.read()
            if st.button("📥 Index this PDF"):
                with st.spinner(f"Reading and indexing '{uploaded_file.name}'..."):
                    saved_path = pdf_service.save_upload(file_bytes, uploaded_file.name)
                    chunk_count = pdf_agent.ingest_pdf(saved_path)
                    pdf_agent.reset_memory()
                st.session_state.pdf_indexed = True
                st.success(f"✅ Indexed {chunk_count} chunks from '{uploaded_file.name}'.")
        except (ValidationError, PDFServiceError, PDFAgentError, VectorServiceError, EmbeddingServiceError) as exc:
            st.error(f"❌ {exc}")

    st.divider()
    st.markdown("### 💬 Ask a question about the uploaded PDF")

    for turn in st.session_state.pdf_chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question_input = st.chat_input("Ask something about the document...")
    if question_input:
        try:
            question = validate_question(question_input)
        except ValidationError as exc:
            st.error(f"⚠️ {exc}")
            return

        st.session_state.pdf_chat.append({"role": "user", "content": question})
        db.add_chat_message(st.session_state.session_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching document and thinking..."):
                try:
                    result = pdf_agent.ask(question)
                    st.markdown(result.answer)
                    if result.confidence_note:
                        st.caption(f"⚠️ {result.confidence_note}")
                    st.session_state.pdf_chat.append({"role": "assistant", "content": result.answer})
                    db.add_chat_message(st.session_state.session_id, "assistant", result.answer)
                except (PDFAgentError, LLMServiceError, VectorServiceError) as exc:
                    error_msg = f"❌ {exc}"
                    st.error(error_msg)
                    st.session_state.pdf_chat.append({"role": "assistant", "content": error_msg})


# ==============================================================================
# Chat history page
# ==============================================================================
def render_history_tab(db: SQLiteDatabase) -> None:
    st.subheader("🕓 Chat & Notes History")
    session_id = st.session_state.session_id

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Conversation History")
        history = db.get_chat_history(session_id)
        if not history:
            st.info("No conversation history yet for this session.")
        for msg in history:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
        if history and st.button("🗑️ Clear chat history"):
            db.clear_chat_history(session_id)
            st.session_state.pdf_chat = []
            st.rerun()

    with col2:
        st.markdown("#### Saved Study Notes")
        topics = db.get_all_topics(session_id)
        if not topics:
            st.info("No study notes saved yet.")
        for topic in topics:
            with st.expander(f"📁 {topic}"):
                for note in db.get_notes(session_id, topic=topic):
                    st.markdown(f"**{note.note_type} ({note.level})** — {note.created_at}")
                    st.text(note.content[:300] + ("..." if len(note.content) > 300 else ""))


# ==============================================================================
# Main entrypoint
# ==============================================================================
def main() -> None:
    inject_dark_theme()
    init_session_state()
    load_theme()
    init_session_state()

    st.sidebar.title(f"🎓 {settings.app_name}")
    st.sidebar.caption(f"Session: `{st.session_state.session_id[:8]}...`")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model:** " + settings.gemini_model + "\n\n"
        "**Embeddings:** " + settings.embedding_model_name + "\n\n"
        "Built with LangChain, FAISS, Streamlit."
    )

    try:
        db = get_database()

        (
            study_agent,
            interview_agent,
            quiz_agent,
            mock_interview_agent,
            pdf_agent,
            pdf_service,
        ) = get_agents()

    except (
        LLMServiceError,
        EmbeddingServiceError,
        VectorServiceError,
    ) as exc:
        st.error(f"❌ Failed to initialize application: {exc}")
        logger.exception("Fatal initialization error")
        st.stop()

    render_dashboard(
    db,
    st.session_state.session_id,
)

    tab_study, tab_pdf, tab_history, tab_mock = st.tabs([
        "📘 Study Generator",
        "📄 PDF Q&A",
        "🕓 History",
        "🎤 Mock Interview",
    ])

    with tab_study:
        render_study_tab(
            study_agent,
            interview_agent,
            quiz_agent,
            db,
        )

    with tab_pdf:
        render_pdf_tab(
            pdf_agent,
            pdf_service,
            db,
        )

    with tab_history:
        render_history_tab(db)

    with tab_mock:

        st.subheader("🎤 AI Mock Interview")

        topic = st.text_input(
            "Interview Topic",
            placeholder="e.g. Java, Python, SQL",
            key="mock_topic",
        )

        difficulty = st.selectbox(
            "Difficulty",
            ["beginner", "intermediate", "advanced"],
            key="mock_level",
        )

        if st.button("🎤 Start Interview"):

            question = mock_interview_agent.generate_question(
                topic,
                difficulty,
            )

            st.session_state.mock_question = question

        if "mock_question" in st.session_state:

            st.info(st.session_state.mock_question)

            answer = st.text_area(
                "Your Answer",
                height=200,
                key="mock_answer",
            )

            if st.button("Submit Answer"):

                result = mock_interview_agent.evaluate_answer(
                    topic,
                    st.session_state.mock_question,
                    answer,
                )
                db.save_mock_interview(
    session_id=st.session_state.session_id,
    topic=topic,
    question=st.session_state.mock_question,
    answer=answer,
    score=str(result["score"]),
)

                st.success(f"Score: {result['score']}/10")

                st.markdown("### Strengths")
                for s in result["strengths"]:
                    st.write("✅", s)

                st.markdown("### Weaknesses")
                for w in result["weaknesses"]:
                    st.write("❌", w)

                st.markdown("### Ideal Answer")
                st.write(result["ideal_answer"])

if __name__ == "__main__":
    main()
