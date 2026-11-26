
import os
import time
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from ddgs import DDGS

app = FastAPI(
    title="Affiliate Intelligence",
    description="AI-powered affiliate market research platform",
    version="5.0.0"
)

MODEL_NAME = "gemini-3.5-flash-lite"

DEFAULT_NICHE = "Discover promising affiliate opportunities across all product categories"


# ==========================================
# REQUEST MODELS
# ==========================================

class ResearchRequest(BaseModel):
    niche: str = DEFAULT_NICHE
    number_of_programs: int = Field(default=10, ge=1, le=10)


class AIRequest(BaseModel):
    prompt: str


class AffiliateAnalysisRequest(BaseModel):
    company_name: str
    product_name: str
    product_description: str = "Not provided"
    affiliate_details: str = "Not verified"
    target_audience: str = "Not specified"


# ==========================================
# GEMINI CONNECTION
# ==========================================

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing in Railway."
        )

    return genai.Client(api_key=api_key)


def generate_ai_response(prompt: str):
    client = get_gemini_client()

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if not response.text:
                raise Exception("Gemini returned an empty response.")

            return response.text

        except Exception as error:
            error_message = str(error)
            print(f"Gemini error: {error_message}")

            if "429" in error_message:
                raise HTTPException(
                    status_code=429,
                    detail="Gemini quota or rate limit reached. Try again later."
                )

            if "503" in error_message:
                if attempt == 0:
                    time.sleep(3)
                    continue

                raise HTTPException(
                    status_code=503,
                    detail="Gemini is temporarily unavailable."
                )

            raise HTTPException(
                status_code=502,
                detail="Gemini request failed. Check Railway logs."
            )


# ==========================================
# WEB RESEARCH ENGINE
# ==========================================

def search_web(query: str, max_results: int = 4):
    try:
        results = DDGS().text(
            query,
            max_results=max_results,
            region="us-en",
            safesearch="moderate"
        )

        return list(results or [])

    except Exception as error:
        print(f"Web search error for '{query}': {error}")
        return []


def collect_market_research(niche: str):
    """
    Discover affiliate opportunities across multiple
    product categories using public web search.
    """

    search_queries = [
        "best affiliate programs and products 2026",
        "SaaS software affiliate programs recurring commissions",
        "physical products ecommerce affiliate programs",
        "digital products online courses affiliate programs",
        "AI tools and software affiliate programs",
        "consumer technology gadgets affiliate programs",
        "home lifestyle products affiliate programs"
    ]

    if niche and niche != DEFAULT_NICHE:
        search_queries.insert(
            0,
            f"{niche} products and affiliate programs"
        )

    collected_results = []
    seen_urls = set()

    for query in search_queries:
        results = search_web(query, max_results=4)

        for item in results:
            url = item.get("href") or item.get("url")

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            collected_results.append({
                "search_query": query,
                "title": item.get("title", ""),
                "url": url,
                "description": item.get("body", "")
            })

    return collected_results


# ==========================================
# AI MARKET RESEARCH AGENT
# ==========================================

def run_market_research(request: ResearchRequest):

    web_results = collect_market_research(request.niche)

    if not web_results:
        raise HTTPException(
            status_code=503,
            detail=(
                "Web search returned no results. "
                "Please try again later."
            )
        )

    research_data = json.dumps(
        web_results,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
You are the Market Research Agent of Affiliate Intelligence.

Your mission is to discover promising affiliate business
opportunities across different product categories.

RESEARCH SCOPE:
{request.niche}

NUMBER OF OPPORTUNITIES:
Up to {request.number_of_programs}

LIVE WEB SEARCH RESULTS:
{research_data}

YOUR TASKS:

1. DISCOVER
Identify companies, products, services and affiliate
opportunities appearing in the supplied search results.

2. CLASSIFY
Classify each opportunity into categories such as:
- AI tools
- SaaS and software
- Physical products
- Consumer technology
- Digital products
- Online education
- Home and lifestyle
- Other relevant categories

3. EVALUATE
Evaluate each opportunity based on:
- Potential market demand
- Target audience
- Product usefulness
- Competition
- Marketing difficulty
- Potential affiliate monetization
- Possibility of recurring revenue
- Content creation potential
- Business risks

4. VERIFY
Use only the supplied search results as research evidence.

Do not claim that you visited a website.
Do not claim that an affiliate program is currently open
unless the supplied evidence explicitly confirms it.

Never invent:
- Commission percentages
- Cookie durations
- Affiliate program terms
- Product prices
- Customer numbers
- Market statistics
- Sources or URLs

If information is missing, write "Not verified".

5. SCORE
Give each opportunity a score from 1 to 10.

Explain the reasoning behind every score.
Scores are preliminary estimates, not measured facts.

6. RANK
Rank the strongest opportunities.

7. RECOMMEND
Select the best three opportunities for deeper research.

For every opportunity include:
- Company
- Product
- Category
- Official website, if present in search results
- Target audience
- Evidence found
- Affiliate program status
- Commission information, if verified
- Recurring revenue potential
- Advantages
- Disadvantages
- Marketing potential
- Risks
- Opportunity score
- Sources with URLs

IMPORTANT:
A search result mentioning a product is not proof
that its affiliate program exists or is accepting applications.

Separate verified evidence, estimates and unknown information.

Finish with a strategic recommendation explaining
which opportunities deserve further investigation.

Return a structured professional report.
"""

    analysis = generate_ai_response(prompt)

    return {
        "status": "success",
        "agent": "Market Research Agent",
        "model": MODEL_NAME,
        "search_results_found": len(web_results),
        "live_web_search": True,
        "research_report": analysis,
        "source_results": web_results
    }


# ==========================================
# MAIN ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "Affiliate Intelligence",
        "version": "5.0.0",
        "ai_model": MODEL_NAME,
        "web_research": "enabled"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ai-test")
def ai_test():

    prompt = """
    Introduce yourself as the AI assistant
    of Affiliate Intelligence.

    Explain your main responsibilities in two short sentences.
    """

    result = generate_ai_response(prompt)

    return {
        "status": "success",
        "model": MODEL_NAME,
        "response": result
    }


# ==========================================
# AUTONOMOUS MARKET DISCOVERY
# ==========================================

@app.post("/research")
def research(request: ResearchRequest):

    return run_market_research(request)


# ==========================================
# GENERAL AI ASSISTANT
# ==========================================

@app.post("/ask-ai")
def ask_ai(request: AIRequest):

    result = generate_ai_response(request.prompt)

    return {
        "status": "success",
        "model": MODEL_NAME,
        "response": result
    }


# ==========================================
# AFFILIATE ANALYST
# ==========================================

@app.post("/analyze-affiliate")
def analyze_affiliate(request: AffiliateAnalysisRequest):

    prompt = f"""
You are the Affiliate Program Analyst of Affiliate Intelligence.

Analyze this opportunity:

COMPANY: {request.company_name}
PRODUCT: {request.product_name}
DESCRIPTION: {request.product_description}
AFFILIATE DETAILS: {request.affiliate_details}
TARGET AUDIENCE: {request.target_audience}

Evaluate:

1. Product value and market demand.
2. Target audience.
3. Affiliate commission model.
4. Recurring revenue potential.
5. Cookie duration and program conditions.
6. Competition.
7. Marketing difficulty.
8. Advantages and disadvantages.
9. Marketing channels.
10. Business risks.
11. Overall opportunity score from 1 to 10.

RULES:

- Do not invent commission percentages.
- Do not invent cookie durations.
- Do not claim unverified details are facts.
- Distinguish facts from estimates.
- Do not promise earnings.
- Mark missing information as Not verified.

Finish with:
A. Overall assessment.
B. Recommendation: High, Medium or Low potential.
C. Information requiring verification.
D. Recommended next steps.

Return a structured professional report.
"""

    result = generate_ai_response(prompt)

    return {
        "status": "success",
        "agent": "Affiliate Analyst",
        "model": MODEL_NAME,
        "company": request.company_name,
        "product": request.product_name,
        "analysis": result
    }


# ==========================================
# FULL RESEARCH AND ANALYSIS WORKFLOW
# ==========================================

@app.post("/full-workflow")
def full_workflow(request: ResearchRequest):

    research_result = run_market_research(request)

    analysis_prompt = f"""
You are the Senior Affiliate Strategy Analyst
of Affiliate Intelligence.

Review the following live-search-based research report:

{research_result["research_report"]}

Your mission is to independently evaluate the
identified opportunities and create a final shortlist.

Evaluate:

1. Market demand evidence.
2. Product value.
3. Target audience.
4. Affiliate program evidence.
5. Commission potential.
6. Recurring revenue potential.
7. Competition.
8. Marketing difficulty.
9. Content creation opportunities.
10. Business risks.

Do not invent facts, commissions, cookie durations,
sources or URLs.

Do not treat preliminary opportunity scores
as verified market data.

Provide:

- Ranked list of opportunities.
- Best three opportunities.
- Reasons for selecting them.
- Missing information to verify.
- Recommended marketing channels.
- Suggested next research actions.

Clearly distinguish evidence from estimates.

Return a professional strategic report.
"""

    final_analysis = generate_ai_response(analysis_prompt)

    return {
        "status": "success",
        "workflow": "Live Research to Strategic Recommendation",
        "model": MODEL_NAME,
        "niche": request.niche,
        "search_results_found": research_result["search_results_found"],
        "research_report": research_result["research_report"],
        "final_analysis": final_analysis,
        "source_results": research_result["source_results"]
    }
