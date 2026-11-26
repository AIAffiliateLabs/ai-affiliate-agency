
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

app = FastAPI(
    title="AI Affiliate Agency",
    description="AI-powered affiliate marketing automation platform",
    version="2.0.0"
)

class AIRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AI Affiliate Agency",
        "version": "2.0.0",
        "message": "Your AI Affiliate Agency is running!"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ai-test")
def ai_test():
    return ask_gemini(
        "Introduce yourself as the AI assistant of an affiliate marketing agency."
    )

@app.post("/ask-ai")
def ask_ai(request: AIRequest):
    return ask_gemini(request.prompt)

def ask_gemini(prompt: str):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing in Railway variables."
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return {
            "status": "success",
            "model": "Gemini",
            "response": response.text
        }

    except Exception as error:
        print(f"Gemini error: {error}")
        raise HTTPException(
            status_code=502,
            detail="Gemini connection failed. Check Railway deployment logs."
        )
