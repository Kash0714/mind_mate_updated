import streamlit as st
import os
import time
import random
import base64
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

st.set_page_config(page_title="💙 Youth Mental Wellness AI", layout="wide")
MAX_HISTORY = 10

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY")
    st.stop()

with st.sidebar:
    st.header("🧠 Model")
    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
    )

    st.markdown("---")
    st.header("🧘 Quick Tip")
    st.info(random.choice([
        "Take a deep breath 🌿",
        "Drink water 💧",
        "Go for a walk 🚶",
        "Talk to someone 🤝",
        "Small steps matter 🚀"
    ]))

model = ChatGroq(model=model_name, api_key=groq_api_key)

BLOCKLIST = ["ignore previous instructions", "you are now", "act as"]

def sanitize(text):
    text = text[:500]
    for w in BLOCKLIST:
        if w in text.lower():
            st.warning("⚠️ Suspicious input detected")
            return ""
    return text

# ================= GMAIL =================
def get_credentials():
    try:
        t = st.secrets["gmail_token"]
    except:
        return None

    creds = Credentials(
        token=t["token"],
        refresh_token=t["refresh_token"],
        token_uri=t["token_uri"],
        client_id=t["client_id"],
        client_secret=t["client_secret"],
        scopes=t["scopes"],
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        st.warning("Token refreshed")

    return creds

class EmailInput(BaseModel):
    to: str
    subject: str
    body: str

def send_email(to: str, subject: str, body: str):
    try:
        creds = get_credentials()
        if not creds:
            return "No Gmail configured"

        service = build("gmail", "v1", credentials=creds)

        msg = base64.urlsafe_b64encode(
            f"To:{to}\nSubject:{subject}\n\n{body}".encode()
        ).decode()

        service.users().messages().send(
            userId="me",
            body={"raw": msg}
        ).execute()

        with open("crisis_log.txt", "a") as f:
            f.write(f"{datetime.now()} - {body}\n")

        return "Email sent"

    except HttpError as e:
        return f"Gmail error {e.status_code}"
    except Exception as e:
        return str(e)

send_email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email_tool",
    description="Send crisis email",
    args_schema=EmailInput
)

if "supervisor" not in st.session_state:

    voice = create_react_agent(
        model=model,
        tools=[],
        prompt="You are a friendly companion.",
        name="voice"
    )

    therapist = create_react_agent(
        model=model,
        tools=[],
        prompt="Help with stress and anxiety calmly.",
        name="therapist"
    )

    motivational = create_react_agent(
        model=model,
        tools=[],
        prompt="Motivate the user positively.",
        name="motivational"
    )

    crisis = create_react_agent(
    model=model,
    tools=[send_email_tool],
    prompt="""
    User is in emotional crisis.

    1. First respond with deep empathy
    2. Then send an email using send_email_tool
    3. Use parent's email for sending alert

    Be supportive and calm.
    """,
    name="crisis"
)

    memory = MemorySaver()

    st.session_state.supervisor = create_supervisor(
        agents=[voice, therapist, motivational, crisis],
        model=model,
        prompt="""
        Route:
        suicide → crisis
        stress → therapist
        fail → motivational
        else → voice
        """
    ).compile(checkpointer=memory)

if "user" not in st.session_state:

    with st.form("form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        parent = st.text_input("Parent Email")

        if st.form_submit_button("Start"):
            if not all([name, email, parent]):
                st.error("Fill all fields")
            else:
                bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    bar.progress(i+1)

                st.session_state.user = {
                    "name": name,
                    "email": email,
                    "parent": parent
                }

                st.rerun()

if "user" in st.session_state:

    st.title(f"Welcome {st.session_state.user['name']}")

    if prompt := st.chat_input("Talk..."):

        prompt = sanitize(prompt)
        if not prompt:
            st.stop()

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                response = st.session_state.supervisor.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": st.session_state.user["email"]}}
                )

                ai_response = response["messages"][-1].content

                st.markdown(ai_response)

    if st.sidebar.button("Reset"):
        st.session_state.clear()
        st.rerun()