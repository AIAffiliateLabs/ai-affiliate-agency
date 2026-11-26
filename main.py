
import os
import time
import json
import re
import ipaddress
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone

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
    version="7.1.0"
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
    """
    Call Gemini with bounded exponential backoff for transient 503/429 errors.
    The function never switches models or enables billing automatically.
    """
    client = get_gemini_client()
    max_attempts = 4
    last_error = None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text

        except Exception as error:
            last_error = error
            error_message = str(error)
            print(
                f"Gemini error on attempt {attempt + 1}/{max_attempts}: "
                f"{error_message}"
            )

            is_unavailable = "503" in error_message or "UNAVAILABLE" in error_message
            is_rate_limited = "429" in error_message or "RESOURCE_EXHAUSTED" in error_message

            if is_unavailable or is_rate_limited:
                if attempt < max_attempts - 1:
                    # 3s, 6s, 12s; bounded to avoid long request hangs.
                    wait_seconds = min(3 * (2 ** attempt), 12)
                    print(f"Retrying Gemini request in {wait_seconds} seconds.")
                    time.sleep(wait_seconds)
                    continue

                if is_rate_limited:
                    raise HTTPException(
                        status_code=429,
                        detail=(
                            "Gemini rate limit or free-tier quota was reached "
                            "after automatic retries. Wait and try again. "
                            "No billing changes were made."
                        )
                    )

                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Gemini is experiencing high demand and remained "
                        "unavailable after automatic retries. Please try again "
                        "later. Completed agent results, if any, are preserved "
                        "in the workflow error response."
                    )
                )

            raise HTTPException(
                status_code=502,
                detail=(
                    "Gemini request failed for a non-retryable reason. "
                    "Check Railway logs for the underlying error."
                )
            )

    raise HTTPException(
        status_code=503,
        detail=f"Gemini request failed after retries: {str(last_error)[:250]}"
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
# SAFE OFFICIAL-WEBSITE INSPECTION
# ==========================================

MAX_PAGE_BYTES = 450_000
PAGE_TIMEOUT_SECONDS = 8
MAX_INITIAL_PAGES = 10
MAX_AFFILIATE_LINKS_PER_DOMAIN = 2

AFFILIATE_TERMS = (
    "affiliate", "partner", "partnership", "referral",
    "commission", "ambassador", "creator"
)


def _validate_public_http_url(url: str):
    """Allow only public HTTP(S) destinations; block local/private targets."""
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only public HTTP/HTTPS URLs are allowed.")

    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Local hostnames are not allowed.")

    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError("Hostname could not be resolved.") from exc

    if not addresses:
        raise ValueError("Hostname has no public address.")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("Private, reserved or non-public IP destinations are blocked.")

    return parsed


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        absolute = urljoin(req.full_url, newurl)
        _validate_public_http_url(absolute)
        return super().redirect_request(req, fp, code, msg, headers, absolute)


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts = []
        self.text_parts = []
        self.links = []
        self.in_title = False
        self.skip_depth = 0
        self.meta_description = ""

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        if tag in ("script", "style", "noscript", "svg"):
            self.skip_depth += 1
        if tag == "meta" and attrs.get("name", "").lower() == "description":
            self.meta_description = attrs.get("content", "")
        if tag == "a" and attrs.get("href"):
            self.links.append((attrs["href"], attrs.get("aria-label", "")))

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag in ("script", "style", "noscript", "svg") and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self.in_title:
            self.title_parts.append(clean)
        self.text_parts.append(clean)


def inspect_public_page(url: str):
    """Fetch a small amount of public HTML without executing scripts."""
    try:
        _validate_public_http_url(url)
        request = Request(
            url,
            headers={
                "User-Agent": "AffiliateIntelligenceResearchBot/1.0 (+public-page-inspection)",
                "Accept": "text/html,application/xhtml+xml"
            }
        )
        opener = build_opener(_SafeRedirectHandler())
        with opener.open(request, timeout=PAGE_TIMEOUT_SECONDS) as response:
            final_url = response.geturl()
            _validate_public_http_url(final_url)
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type and "xhtml" not in content_type:
                return {
                    "requested_url": url,
                    "final_url": final_url,
                    "status": "skipped_non_html",
                    "reason": "Response was not HTML."
                }
            raw = response.read(MAX_PAGE_BYTES + 1)
            if len(raw) > MAX_PAGE_BYTES:
                raw = raw[:MAX_PAGE_BYTES]
            charset = response.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")

        parser = _PageParser()
        parser.feed(html)
        page_text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
        return {
            "requested_url": url,
            "final_url": final_url,
            "status": "inspected",
            "title": " ".join(parser.title_parts)[:250],
            "description": parser.meta_description[:500],
            "text_excerpt": page_text[:6500],
            "links": parser.links[:250]
        }
    except Exception as exc:
        return {
            "requested_url": url,
            "status": "fetch_failed",
            "reason": str(exc)[:300]
        }


def collect_official_site_evidence(search_results):
    """
    Inspect a bounded set of search-result pages and affiliate-related links.
    A fetched page is evidence, not automatic proof that a domain is official.
    """
    inspected = []
    seen = set()
    candidates = []

    for item in search_results:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append(url)
        if len(candidates) >= MAX_INITIAL_PAGES:
            break

    for url in candidates:
        page = inspect_public_page(url)
        inspected.append(page)

    affiliate_candidates = []
    for page in inspected:
        if page.get("status") != "inspected":
            continue
        base = page.get("final_url") or page.get("requested_url")
        base_host = urlparse(base).hostname
        for href, anchor in page.get("links", []):
            combined = f"{href} {anchor}".lower()
            if any(term in combined for term in AFFILIATE_TERMS):
                candidate = urljoin(base, href)
                if urlparse(candidate).hostname == base_host and candidate not in seen:
                    seen.add(candidate)
                    affiliate_candidates.append(candidate)

    # Keep the follow-up inspection small and same-domain.
    for url in affiliate_candidates[:MAX_AFFILIATE_LINKS_PER_DOMAIN * 5]:
        inspected.append(inspect_public_page(url))

    # Remove raw link lists from the evidence sent to Gemini to keep prompts compact.
    compact = []
    for page in inspected:
        item = dict(page)
        item.pop("links", None)
        compact.append(item)

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages_inspected": len(compact),
        "pages": compact
    }


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

def run_verification_agent(discovery_report, web_results, official_site_evidence):

    research_data = json.dumps(web_results, ensure_ascii=False, indent=2)
    inspected_data = json.dumps(official_site_evidence, ensure_ascii=False, indent=2)

    prompt = f"""
You are the Verification Agent of Affiliate Intelligence, version 7.

Your job is to assess each opportunity using BOTH search evidence and
the public website pages that the system actually fetched.

DISCOVERY REPORT:
{discovery_report}

SEARCH EVIDENCE:
{research_data}

DIRECTLY INSPECTED PUBLIC PAGE EVIDENCE:
{inspected_data}

For every opportunity, report:
1. Company and product/service.
2. Candidate official domain and why it appears to belong to the company.
3. Whether an affiliate/partner/referral page was directly inspected.
4. Evidence of program availability or application status.
5. Commission rate/model, only if explicitly shown.
6. Cookie duration, only if explicitly shown.
7. Important eligibility, geography, payout or program terms, only if shown.
8. Source URLs and short exact evidence excerpts.
9. What remains unknown or contradictory.

Use exactly one of these verification statuses:
- Official affiliate page inspected — program evidence found
- Official website inspected — affiliate program not found in inspected pages
- Third-party evidence only
- Not verified
- Conflicting evidence

CRITICAL EVIDENCE RULES:
- A fetched page is not automatically an official page. Assess the domain carefully.
- A search snippet is not direct website verification.
- Do not say a program is active or accepting applications unless the inspected
  page explicitly supports that conclusion.
- Do not infer commissions, cookie durations, payouts or terms.
- If a value is absent, write "Not stated on inspected pages".
- If a page could not be fetched, say so.
- "Not found in inspected pages" does NOT mean the program does not exist.
- Do not invent URLs, quotes, dates or facts.
- Keep facts, interpretation and unknowns clearly separated.

For each opportunity include:
- Company / product
- Candidate official website
- Affiliate page URL (if actually present in inspected evidence)
- Direct inspection result
- Program/application status
- Commission
- Cookie duration
- Terms and eligibility
- Evidence excerpts with source URLs
- Verification status
- Missing information / next checks

Finish with a ranked shortlist based on strength of evidence, not estimated earnings.
Return a structured professional report.
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
    """
    Run agents sequentially and preserve completed outputs if a later agent fails.
    """
    web_results = collect_market_research(request.niche)

    if not web_results:
        raise HTTPException(
            status_code=503,
            detail="Web search returned no results. Please try again later."
        )

    completed = {}
    current_agent = "Market Discovery Agent"

    try:
        discovery = run_discovery_agent(request, web_results)
        completed["market_discovery"] = {
            "status": "completed",
            "report": discovery
        }

        current_agent = "Verification Agent"
        official_site_evidence = collect_official_site_evidence(web_results)
        verification = run_verification_agent(
            discovery,
            web_results,
            official_site_evidence
        )
        completed["verification"] = {
            "status": "completed",
            "report": verification
        }

        current_agent = "Competition Analysis Agent"
        competition = run_competition_agent(discovery, verification)
        completed["competition"] = {
            "status": "completed",
            "report": competition
        }

        current_agent = "Financial Analyst Agent"
        financial = run_financial_agent(discovery, verification)
        completed["financial_analyst"] = {
            "status": "completed",
            "report": financial
        }

        current_agent = "Chief Strategy Agent"
        strategy = run_strategy_agent(
            discovery,
            verification,
            competition,
            financial
        )
        completed["chief_strategy"] = {
            "status": "completed",
            "report": strategy
        }

    except HTTPException as error:
        # Return HTTP 200 with explicit partial status so completed work is not lost.
        return {
            "status": "partial",
            "project": "Affiliate Intelligence",
            "version": "7.1.0",
            "workflow": "Five-Agent Affiliate Intelligence System",
            "model": MODEL_NAME,
            "niche": request.niche,
            "search_results_found": len(web_results),
            "failed_agent": current_agent,
            "error": {
                "status_code": error.status_code,
                "detail": error.detail
            },
            "completed_agents": completed,
            "source_results": web_results
        }

    except Exception as error:
        print(f"Unexpected workflow error in {current_agent}: {error}")
        return {
            "status": "partial",
            "project": "Affiliate Intelligence",
            "version": "7.1.0",
            "workflow": "Five-Agent Affiliate Intelligence System",
            "model": MODEL_NAME,
            "niche": request.niche,
            "search_results_found": len(web_results),
            "failed_agent": current_agent,
            "error": {
                "status_code": 500,
                "detail": "Unexpected agent error. Check Railway logs."
            },
            "completed_agents": completed,
            "source_results": web_results
        }

    return {
        "status": "success",
        "project": "Affiliate Intelligence",
        "version": "7.1.0",
        "workflow": "Five-Agent Affiliate Intelligence System",
        "model": MODEL_NAME,
        "niche": request.niche,
        "search_results_found": len(web_results),
        "agents": completed,
        "source_results": web_results,
        "official_site_evidence": official_site_evidence
    }


# ==========================================
# MAIN ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "Affiliate Intelligence",
        "version": "7.1.0",
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
