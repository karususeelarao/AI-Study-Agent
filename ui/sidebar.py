import streamlit as st


def render_sidebar():

    with st.sidebar:

        st.markdown("# 🎓 AI Study Assistant")

        st.caption("Powered by Gemini AI")

        st.divider()

        st.subheader("📚 Learning")

        st.button("🏠 Dashboard", use_container_width=True)

        st.button("📘 Study Hub", use_container_width=True)

        st.button("🎤 Interview Hub", use_container_width=True)

        st.button("❓ Quiz Center", use_container_width=True)

        st.button("📄 PDF Intelligence", use_container_width=True)

        st.button("🗂 Notes Library", use_container_width=True)

        st.button("💬 Chat History", use_container_width=True)

        st.button("📈 Analytics", use_container_width=True)

        st.button("⚙ Settings", use_container_width=True)

        st.divider()

        st.caption("System")

        st.success("🟢 Gemini Connected")

        st.success("🟢 SQLite")

        st.success("🟢 FAISS")