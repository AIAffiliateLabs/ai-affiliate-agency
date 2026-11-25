
from fastapi import FastAPI

app = FastAPI(
    title="AI Affiliate Agency",
    description="AI-powered affiliate marketing automation platform",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "AI Affiliate Agency",
        "message": "Our AI platform is starting!"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
