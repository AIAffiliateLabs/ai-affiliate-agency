
import os
import time
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from ddgs import DDGS


# ==========================================
# APPLICATION SETTINGS
# ==========================================

app = FastAPI(
    title="Affiliate Intelligence",
    description="AI-powered autonomous affiliate research platform",
    version="6.0.0"
)

MODEL_NAME = "gemini-3.5-flash-lite"

DEFAULT_NICHE = (
    "Discover promising affiliate opportunities "
    "across all product categories"
)


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
                    detail=(
                        "Gemini quota or rate limit reached. "
                        "Try again later."
                    )
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
    Search across multiple product categories.
    Search snippets are research leads, not verified facts.
    """

    search_queries = [
        "affiliate programs official websites 2026",
        "SaaS software affiliate programs official",
        "physical products ecommerce affiliate programs",
        "digital products online courses affiliate programs",
        "AI tools software affiliate programs",
        "consumer technology products affiliate programs",
        "home lifestyle products affiliate programs",
        "online services affiliate partnership programs"
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
# AGENT 1 — MARKET DISCOVERY
# ==========================================

def run_discovery_agent(request, web_results):

    research_data = json.dumps(
        web_results,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
You are the Market Discovery Agent of Affiliate Intelligence.

Your mission is to independently discover promising
affiliate business opportunities.

RESEARCH SCOPE:
{request.niche}

MAXIMUM OPPORTUNITIES:
{request.number_of_programs}

PUBLIC WEB SEARCH RESULTS:
{research_data}

YOUR RESPONSIBILITIES:

1. Discover companies, products and services.
2. Identify different product categories.
3. Identify potential customer audiences.
4. Evaluate product usefulness and demand signals.
5. Identify possible affiliate monetization models.
6. Identify recurring revenue possibilities.
7. Evaluate content creation opportunities.
8. Identify business risks.
9. Give each opportunity a preliminary score from 1 to 10.
10. Rank the strongest opportunities.

Explore different industries. Do not focus only on AI
or software.

Consider physical products, digital products,
SaaS, education, technology, services and other markets.

For every opportunity include:
- Company
- Product or service
- Category
- Website URL, if supplied
- Target audience
- Evidence from search results
- Possible affiliate model
- Recurring revenue potential
- Advantages
- Disadvantages
- Marketing opportunities
- Risks
- Preliminary opportunity score
- Missing information

IMPORTANT:

Search results are leads, not proof of an active
affiliate program.

Never invent commission rates, cookie durations,
prices, customer numbers or program conditions.

Mark missing information as "Not verified".

Clearly separate evidence from estimates.

Finish with the best opportunities for further analysis.
Return a structured professional report.
"""

    return generate_ai_response(prompt)


# ==========================================
# AGENT 2 — VERIFICATION
# ==========================================

def run_verification_agent(discovery_report, web_results):

    research_data = json.dumps(
        web_results,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
You are the Verification Agent of Affiliate Intelligence.

Your job is to critically examine the opportunities
identified by the Market Discovery Agent.

DISCOVERY REPORT:
{discovery_report}

AVAILABLE SEARCH EVIDENCE:
{research_data}

For every identified opportunity, examine:

1. Is the company identifiable from the evidence?
2. Is the product or service identifiable?
3. Is there evidence of an affiliate program?
4. Is there evidence that the program is currently accepting
   applications?
5. Is there evidence of commission conditions?
6. Is there evidence of cookie duration?
7. Is the supplied website likely to be the official website?
8. What information is missing?
9. What should be checked directly on the official website?

Use these evidence labels:

- Supported by search evidence
- Partially supported
- Not verified
- Conflicting evidence

CRITICAL RULES:

You have search-result information only.
You have not independently opened or inspected
the official websites.

Do not claim that you visited an official website.
Do not claim that an affiliate program is active
unless the supplied evidence explicitly supports it.

A search result mentioning a product is not proof
of an active affiliate program.

Never invent commissions, cookie durations,
program terms or application status.

For each opportunity provide:
- Company and product
- Evidence found
- Affiliate program evidence
- Commission evidence
- Cookie duration evidence
- Verification status
- Missing information
- Recommended verification steps

Finish with a list of opportunities that need
the least additional verification.

Return a structured verification report.
"""

    return generate_ai_response(prompt)


# ==========================================
# AGENT 3 — COMPETITION ANALYSIS
# ==========================================

def run_competition_agent(discovery_report, verification_report):

    prompt = f"""
You are the Competition Analysis Agent of Affiliate Intelligence.

Your mission is to evaluate the competitive landscape
for the discovered affiliate opportunities.

DISCOVERY REPORT:
{discovery_report}

VERIFICATION REPORT:
{verification_report}

Analyze each opportunity based on:

1. Likely competition level.
2. Established competitors and publishers.
3. SEO difficulty.
4. Potential search intent.
5. Content marketing opportunities.
6. YouTube content opportunities.
7. Instagram and social media opportunities.
8. LinkedIn opportunities where relevant.
9. Potential underserved audiences.
10. Differentiation opportunities.
11. Difficulty of building authority.
12. Long-term content potential.

For each opportunity provide:

- Competition level: Low, Medium or High
- Reason for the assessment
- Potential audience
- Possible content strategy
- SEO opportunities
- Social media opportunities
- Differentiation strategy
- Main competitive risks
- Preliminary marketing attractiveness score from 1 to 10

IMPORTANT:

You do not have verified keyword-volume or competitor
traffic data.

Do not invent search volumes, website traffic,
market shares or competitor revenue.

Clearly label estimates and assumptions.

Finish by ranking the opportunities with the
most attractive competitive conditions.

Return a structured professional report.
"""

    return generate_ai_response(prompt)


# ==========================================
# AGENT 4 — FINANCIAL ANALYST
# ==========================================

def run_financial_agent(discovery_report, verification_report):

    prompt = f"""
You are the Financial Analyst Agent of Affiliate Intelligence.

Your responsibility is to evaluate the financial
potential and risks of each affiliate opportunity.

DISCOVERY REPORT:
{discovery_report}

VERIFICATION REPORT:
{verification_report}

Analyze:

1. Affiliate monetization model.
2. One-time versus recurring commissions.
3. Potential customer lifetime value.
4. Possible recurring revenue.
5. Customer acquisition challenges.
6. Content production requirements.
7. Estimated operating costs.
8. Platform and software costs.
9. Financial risks.
10. Scalability.

For every opportunity include:

- Known commission information
- Unknown commission information
- Possible revenue model
- Recurring revenue potential
- Cost considerations
- Financial advantages
- Financial disadvantages
- Important assumptions
- Financial attractiveness score from 1 to 10

IMPORTANT:

Do not invent commission percentages.
Do not invent product prices.
Do not promise earnings.
Do not present hypothetical revenue as actual revenue.

If commission information is missing,
write "Not verified".

You may provide qualitative financial estimates,
but clearly label them as estimates.

Do not assume that recurring revenue is available
unless the evidence supports it.

Finish by ranking opportunities based on their
potential financial attractiveness and risk.

Return a structured financial report.
"""

    return generate_ai_response(prompt)


# ==========================================
# AGENT 5 — CHIEF STRATEGY
# ==========================================

def run_strategy_agent(
    discovery_report,
    verification_report,
    competition_report,
    financial_report
):

    prompt = f"""
You are the Chief Strategy Agent of Affiliate Intelligence.

You are responsible for making the final strategic decision.

You must compare the work of all four specialized agents.

MARKET DISCOVERY REPORT:
{discovery_report}

VERIFICATION REPORT:
{verification_report}

COMPETITION REPORT:
{competition_report}

FINANCIAL REPORT:
{financial_report}

YOUR MISSION:

Identify the strongest affiliate business opportunities
for building a scalable AI-powered affiliate media agency.

Evaluate each opportunity based on:

1. Market attractiveness.
2. Product usefulness.
3. Strength of available evidence.
4. Affiliate program verification status.
5. Financial potential.
6. Recurring revenue potential.
7. Competition.
8. Content creation potential.
9. Marketing difficulty.
10. Business risks.
11. Scalability.
12. Suitability for a website, YouTube,
    Instagram and LinkedIn strategy.

YOUR FINAL REPORT MUST INCLUDE:

A. EXECUTIVE SUMMARY

Explain the overall market findings.

B. RANKED SHORTLIST

Rank the strongest opportunities and explain why.

C. TOP THREE RECOMMENDATIONS

For each recommendation include:
- Company
- Product
- Category
- Target audience
- Evidence available
- Verification status
- Financial potential
- Competition
- Recommended marketing channels
- Main risks
- Why it deserves attention

D. REJECTED OR LOWER-PRIORITY OPPORTUNITIES

Explain why some opportunities should not
be prioritized.

E. NEXT ACTIONS

Provide a practical action plan for further research.

F. STRATEGIC CONCLUSION

Recommend which opportunities should receive
the next round of investigation.

IMPORTANT:

Do not invent facts or financial data.
Do not treat preliminary scores as verified data.
Do not recommend an opportunity as fully verified
when the verification report says otherwise.

Distinguish facts, estimates and unknown information.

Return a professional executive strategy report.
"""

    return generate_ai_response(prompt)


# ==========================================
# FIVE-AGENT ORCHESTRATOR
# ==========================================

def run_agent_workflow(request: ResearchRequest):

    web_results = collect_market_research(request.niche)

    if not web_results:
        raise HTTPException(
            status_code=503,
            detail=(
                "Web search returned no results. "
                "Please try again later."
            )
        )

    # Agent 1
    discovery = run_discovery_agent(
        request,
        web_results
    )

    # Agent 2
    verification = run_verification_agent(
        discovery,
        web_results
    )

    # Agent 3
    competition = run_competition_agent(
        discovery,
        verification
    )

    # Agent 4
    financial = run_financial_agent(
        discovery,
        verification
    )

    # Agent 5
    strategy = run_strategy_agent(
        discovery,
        verification,
        competition,
        financial
    )

    return {
        "status": "success",
        "project": "Affiliate Intelligence",
        "version": "6.0.0",
        "workflow": "Five-Agent Affiliate Intelligence System",
        "model": MODEL_NAME,
        "niche": request.niche,
        "search_results_found": len(web_results),

        "agents": {
            "market_discovery": {
                "status": "completed",
                "report": discovery
            },
            "verification": {
                "status": "completed",
                "report": verification
            },
            "competition": {
                "status": "completed",
                "report": competition
            },
            "financial_analyst": {
                "status": "completed",
                "report": financial
            },
            "chief_strategy": {
                "status": "completed",
                "report": strategy
            }
        },

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
        "version": "6.0.0",
        "ai_model": MODEL_NAME,
        "web_research": "enabled",
        "agents": 5
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/agents")
def list_agents():
    return {
        "project": "Affiliate Intelligence",
        "agents": [
            "Market Discovery Agent",
            "Verification Agent",
            "Competition Analysis Agent",
            "Financial Analyst Agent",
            "Chief Strategy Agent"
        ]
    }


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
# MARKET RESEARCH ENDPOINT
# ==========================================

@app.post("/research")
def research(request: ResearchRequest):

    web_results = collect_market_research(request.niche)

    if not web_results:
        raise HTTPException(
            status_code=503,
            detail="Web search returned no results."
        )

    report = run_discovery_agent(
        request,
        web_results
    )

    return {
        "status": "success",
        "agent": "Market Discovery Agent",
        "model": MODEL_NAME,
        "search_results_found": len(web_results),
        "research_report": report,
        "source_results": web_results
    }


# ==========================================
# FIVE-AGENT WORKFLOW ENDPOINT
# ==========================================

@app.post("/agent-workflow")
def agent_workflow(request: ResearchRequest):

    return run_agent_workflow(request)


# Keep the previous endpoint working.
@app.post("/full-workflow")
def full_workflow(request: ResearchRequest):

    return run_agent_workflow(request)


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
