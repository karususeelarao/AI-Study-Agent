import streamlit as st


def render_dashboard(db, session_id):

    # ===========================
    # Hero Section
    # ===========================

    st.markdown("""
# 👋 Welcome Back!

### What would you like to learn today?

AI-powered Study Material • Mock Interviews • PDF Q&A • Quiz Generation
""")

    topic = st.text_input(
        "",
        placeholder="🔍 Ask anything... Java, Python, SQL, React...",
    )

    st.divider()

    # ===========================
    # Load Data
    # ===========================

    topics = db.get_all_topics(session_id)
    notes = db.get_notes(session_id)

    interviews = sum(
        1 for n in notes if n.note_type == "mock_interview"
    )

    quizzes = sum(
        1 for n in notes if n.note_type == "quiz"
    )

    pdfs = sum(
        1 for n in notes if n.note_type == "pdf"
    )

    # ===========================
    # Statistics
    # ===========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(f"""
### 📚 Topics

# {len(topics)}

Topics Studied
""")

    with col2:
        st.success(f"""
### 🎤 Interviews

# {interviews}

Completed
""")

    with col3:
        st.warning(f"""
### ❓ Quizzes

# {quizzes}

Generated
""")

    with col4:
        st.error(f"""
### 📄 PDFs

# {pdfs}

Indexed
""")

    st.divider()

    # ===========================
    # AI Features
    # ===========================

    st.markdown("## ✨ AI Features")

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("### 📘 AI Study Generator")
            st.write(
                "Generate beginner, intermediate and advanced study notes."
            )

    with c2:
        with st.container(border=True):
            st.markdown("### 🎤 Mock Interview")
            st.write(
                "Practice interviews with AI feedback and scoring."
            )

    with c3:
        with st.container(border=True):
            st.markdown("### 📄 PDF Intelligence")
            st.write(
                "Upload PDFs and ask questions using RAG."
            )

    with c4:
        with st.container(border=True):
            st.markdown("### ❓ Quiz Generator")
            st.write(
                "Create MCQs instantly from any topic."
            )

    st.divider()

    # ===========================
    # Recent Activity
    # ===========================

    st.markdown("## 🕒 Recent Activity")

    recent_notes = notes[:5]

    if recent_notes:
        for note in recent_notes:
            st.markdown(f"""
### 📌 {note.topic}

**Type:** `{note.note_type}`

🕒 {note.created_at}

---
""")
    else:
        st.info("No activity yet.")