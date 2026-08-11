import streamlit as st
from streamlit_mic_recorder import mic_recorder


def render_voice_ui():

    st.header("🎙️ AI Voice Assistant")

    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹ Stop Recording",
        key="voice",
    )

    return audio