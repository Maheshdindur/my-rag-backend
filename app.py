import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import requests
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# -- Pushover notification function --
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

# -- RAG context loader --
def load_resume_summary():
    summary_path = "me/summary.txt"
    pdf_path = "me/resume_for_Virtual_Assistant.pdf"
    summary = ""
    resume_text = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read()
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text + "\n"
    return summary, resume_text

summary, resume_text = load_resume_summary()

# -- FastAPI app --
app = FastAPI()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "")
    history = data.get("history", [])
    user_email = data.get("email", "")

    # Compose RAG prompt
    prompt = f"""You are acting as Mahesh Dindur, answering questions on his website.
Your job is to use the following resume and summary to help answer questions about Mahesh's skills, career, and background.

## Summary
{summary}

## Resume
{resume_text}

User message: {user_message}
"""

    # Call Gemini/OpenAI API (using OpenAI-style API for compatibility)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 512,
    }
    try:
        response = requests.post(
            base_url + "chat/completions",
            headers=headers,
            json=payload,
            timeout=20
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer = f"Sorry, there was an error with the AI model: {e}"

    # Record user interest if email provided
    if user_email:
        push(f"User interested: {user_email} | Message: {user_message}")

    # If AI is unsure, record the question
    if "don't know" in answer.lower() or "not sure" in answer.lower():
        push(f"Unknown question recorded: {user_message}")

    return JSONResponse({"answer": answer})

@app.get("/")
def root():
    return {"message": "RAG Chatbot for Mahesh Dindur is running. Use /chat endpoint."}