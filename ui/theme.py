import streamlit as st


def load_theme():

    st.markdown(
        """
<style>

/* Background */

.stApp{
    background:#0F172A;
}

/* Sidebar */

[data-testid="stSidebar"]{
    background:#111827;
    border-right:1px solid #374151;
}

/* Cards */

div[data-testid="stMetric"]{
    background:#1E293B;
    padding:20px;
    border-radius:18px;
    border:1px solid #334155;
}

/* Buttons */

.stButton>button{

    width:100%;

    border-radius:12px;

    border:none;

    background:linear-gradient(90deg,#8B5CF6,#6366F1);

    color:white;

    font-weight:700;

    padding:12px;
}

.stButton>button:hover{

    background:linear-gradient(90deg,#7C3AED,#4F46E5);

}

/* Containers */

.block-container{

    padding-top:2rem;

    max-width:1300px;

}

</style>
""",
        unsafe_allow_html=True,
    )