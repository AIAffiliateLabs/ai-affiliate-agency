
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
    description="AI-powered affiliate research and verification platform",
    version="7.3.0"
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
# CANDIDATE-SPECIFIC VERIFICATION RESEARCH
# ==========================================

def extract_discovery_candidates(discovery_report: str, limit: int = 10):
    """Extract company names from the ranked Markdown comparison table."""
    candidates = []
    seen = set()

    for raw_line in discovery_report.splitlines():
        line = raw_line.strip()
        if not line.startswith('|'):
            continue
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) < 3:
            continue
        rank_text = re.sub(r'[*` ]', '', cells[0])
        if not rank_text.isdigit():
            continue

        name = re.sub(r'[*_`]', '', cells[1]).strip()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'\s*\([^)]*\)\s*$', '', name).strip()

        # Split paired company names, but discard generic descriptors after '/'.
        generic = {
            'high-ticket consumer technology & electronics',
            'high-ticket electronics', 'tech', 'ai tools', 'various tech brands',
            'consumer technology & electronics', 'electronics'
        }
        name_parts = [part.strip() for part in name.split('/') if part.strip()]
        if len(name_parts) > 1:
            name_parts = [part for part in name_parts if part.lower() not in generic]
        else:
            name_parts = [name]

        for candidate_name in name_parts:
            if not candidate_name or candidate_name.lower() in generic:
                continue
            if candidate_name.lower() in seen:
                continue
            seen.add(candidate_name.lower())
            candidates.append({
                'rank': int(rank_text),
                'candidate': candidate_name,
                'category': cells[2] if len(cells) > 2 else 'Not stated'
            })
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break

    return candidates


def collect_candidate_specific_evidence(discovery_report: str, general_search_results):
    """Search each discovered company by name; keep leads tied to that candidate."""
    candidates = extract_discovery_candidates(discovery_report)
    candidate_searches = []
    candidate_results = []
    seen_urls = set()

    for candidate in candidates:
        name = candidate['candidate']
        query = f'"{name}" official website affiliate partner program'
        results = search_web(query, max_results=3)
        cleaned = []

        for item in results:
            url = item.get('href') or item.get('url')
            if not url:
                continue
            record = {
                'candidate': name,
                'search_query': query,
                'title': item.get('title', ''),
                'url': url,
                'description': item.get('body', '')
            }
            cleaned.append(record)
            candidate_results.append(record)
            seen_urls.add(url)

        candidate_searches.append({
            'candidate': name,
            'rank': candidate['rank'],
            'category': candidate['category'],
            'query': query,
            'results_found': len(cleaned),
            'results': cleaned
        })

    # Retain a small general evidence set as context, but prioritize targeted results.
    for item in general_search_results[:5]:
        url = item.get('url')
        if url and url not in seen_urls:
            candidate_results.append({
                'candidate': 'General market context — not attributable to a specific company',
                **item
            })
            seen_urls.add(url)

    return candidates, candidate_searches, candidate_results


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
    """Inspect candidate-specific search pages first, then a small general fallback."""
    inspected = []
    seen = set()
    candidates = []

    # Candidate-specific results are passed first by the workflow.
    for item in search_results:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        candidates.append({"url": url, "candidate": item.get("candidate", "Unassigned")})
        if len(candidates) >= 24:
            break

    for item in candidates:
        page = inspect_public_page(item["url"])
        page["candidate"] = item["candidate"]
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
                    affiliate_candidates.append({
                        "url": candidate,
                        "candidate": page.get("candidate", "Unassigned")
                    })

    # Inspect a bounded number of affiliate/partner links on the same host.
    for item in affiliate_candidates[:15]:
        page = inspect_public_page(item["url"])
        page["candidate"] = item["candidate"]
        inspected.append(page)

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

    broad_scope = request.niche.strip().lower() in {
        DEFAULT_NICHE.lower(),
        "all categories",
        "all affiliate categories",
        "discover all categories",
        "general affiliate opportunities"
    }

    if broad_scope:
        category_strategy = """
BROAD DISCOVERY MODE:
- Compare opportunities across multiple unrelated categories.
- Aim for a balanced shortlist of up to the requested maximum.
- Include SaaS/software, physical consumer products, digital products,
  education/courses, technology, and services where supported by the evidence.
- Do not allow one category to dominate the shortlist without a clear
  evidence-based reason.
- Try to include at least 5 distinct categories when the search evidence
  supports them. Do not invent candidates just to fill a quota.
"""
    else:
        category_strategy = """
FOCUSED DISCOVERY MODE:
- Respect the requested niche and prioritize relevant opportunities.
- You may include adjacent categories only when they are genuinely relevant.
- Do not force unrelated categories into the shortlist.
"""

    prompt = f"""
You are the Market Discovery Agent of Affiliate Intelligence, version 7.3.

MISSION:
Discover, screen, compare and rank promising affiliate business
opportunities using the supplied public web search evidence.

RESEARCH SCOPE:
{request.niche}

MAXIMUM OPPORTUNITIES TO RETURN:
{request.number_of_programs}

DISCOVERY MODE:
{category_strategy}

PUBLIC WEB SEARCH RESULTS (UNVERIFIED LEADS):
{research_data}

REQUIRED WORKFLOW:
1. Extract distinct companies, brands, products and services from the results.
2. Remove duplicates and irrelevant results.
3. Classify each candidate by product category and business model.
4. Assess whether the product solves a clear customer problem.
5. Identify likely target audiences and potential content angles.
6. Look for evidence that an affiliate, referral, partner or creator program
   may exist. A product being popular does not prove it has an affiliate program.
7. Assess recurring-revenue potential only where the business model supports it.
8. Compare candidates using the same scoring criteria.
9. Rank the strongest candidates and explain the trade-offs.

SCORING (1-10, preliminary analytical estimate):
- Product usefulness / problem solved: 20%
- Evidence of demand or market interest: 20%
- Affiliate-program evidence / monetization fit: 20%
- Content and audience potential: 15%
- Recurring-revenue potential: 10%
- Competitive accessibility for a new publisher: 10%
- Evidence quality: 5%

Do not present this score as measured market data or a guarantee of earnings.
Explain briefly why each candidate received its score.

FOR EACH CANDIDATE INCLUDE:
- Rank
- Company / brand
- Product or service
- Category
- Official website URL (only if present in supplied results)
- Target audience
- Customer problem solved
- Search evidence (identify which supplied result supports the point)
- Affiliate-program evidence status: Found in search lead / Not found / Unclear
- Possible monetization model (label as potential unless verified)
- Recurring-revenue potential: High / Medium / Low / Unknown, with reason
- Content opportunities
- Advantages
- Disadvantages and risks
- Preliminary score out of 10 and short rationale
- Key facts still requiring verification

FINAL OUTPUT:
A. Executive summary.
B. Ranked comparison table of the strongest distinct opportunities.
C. Category breakdown (especially in broad discovery mode).
D. Top 3 candidates recommended for the Verification Agent, with reasons.
E. Research gaps and limitations.

STRICT EVIDENCE RULES:
- Search results are leads, not proof of an active affiliate program.
- Never invent commission rates, cookie durations, prices, customer counts,
  conversion rates, program terms, approval requirements or earnings.
- Never claim a program is active unless the supplied evidence explicitly
  supports that status. The Verification Agent will check official sources.
- Mark unavailable information as "Not verified".
- Clearly separate observed evidence, analytical estimates and unknowns.
- If the supplied evidence supports fewer candidates than requested, return
  fewer and explain why. Quality and traceability matter more than quantity.

Return a structured, concise but sufficiently detailed professional report.
"""

    return generate_ai_response(prompt)


# ==========================================
# AGENT 2 — VERIFICATION
# ==========================================

def run_verification_agent(discovery_report, candidate_searches, official_site_evidence):
    """Verify each candidate against its own targeted search and inspected pages."""
    research_data = json.dumps(candidate_searches, ensure_ascii=False, indent=2)
    inspected_data = json.dumps(official_site_evidence, ensure_ascii=False, indent=2)

    prompt = f"""
You are the Verification Agent of Affiliate Intelligence, version 7.3.

MISSION:
Audit each company/product candidate from Market Discovery. Your job is to
verify whether that specific company has an affiliate, referral, partner,
creator or publisher program, using candidate-specific evidence.

DISCOVERY REPORT:
{discovery_report}

CANDIDATE-SPECIFIC SEARCH RESULTS:
{research_data}

PUBLIC PAGES ACTUALLY FETCHED BY THIS SYSTEM:
{inspected_data}

IMPORTANT MATCHING RULE:
- Keep evidence attached to the exact candidate named in each search result/page.
- Do not use a different company's page as proof for this candidate.
- A marketplace/network directory or blog article is third-party evidence,
  not the candidate's official affiliate program page.
- A page is not official merely because it mentions the company. Assess the
  domain and page ownership from the evidence; if uncertain, say so.
- If the candidate is an affiliate software/network platform (for example,
  software used by merchants to run affiliate programs), distinguish it from
  a brand/product that publishers can themselves promote. Do not assume it
  has a publisher-facing affiliate program.

FOR EACH DISCOVERED CANDIDATE, REPORT:
1. Candidate company/product and category.
2. Candidate's likely official domain, only when supported by evidence.
3. Official-domain confidence: High / Medium / Low / Unknown, with reason.
4. Official affiliate/partner page URL, only if it appears in inspected evidence.
5. Whether that page was directly inspected by this system.
6. Program status: explicitly accepting applications / program described but
   application status unclear / not found in inspected pages / unknown.
7. Commission amount/model, only if explicitly stated on a reliable source.
8. Cookie duration, only if explicitly stated.
9. Eligibility, geography, payout threshold and restrictions, only if stated.
10. Short evidence excerpts and their exact source URLs.
11. Unknowns, contradictions and recommended next verification step.

USE EXACTLY ONE VERIFICATION STATUS PER CANDIDATE:
- Official affiliate page inspected — program evidence found
- Official website inspected — affiliate program not found in inspected pages
- Third-party evidence only
- Not verified
- Conflicting evidence

EVIDENCE RULES:
- Search snippets and third-party roundups are leads, not direct verification.
- A fetched page is not automatically official; explain why you consider it
  official or why ownership remains uncertain.
- Do not claim a program is active or accepting applications unless the
  inspected official page explicitly supports that conclusion.
- Do not infer commissions, cookie durations, payouts, eligibility or terms.
- If a value is absent, write "Not stated on inspected pages".
- If fetching failed, say "Page could not be inspected" and include the reason
  when available.
- "Not found in inspected pages" does not mean the program does not exist.
- Never invent URLs, quotes, dates, program status or facts.
- Clearly separate directly observed facts, third-party claims and analysis.

OUTPUT:
A. Executive summary of verification coverage.
B. One verification profile for EVERY candidate in the discovery shortlist.
C. Comparison table with candidate, official domain confidence, status,
   direct affiliate page inspected (Yes/No), commission (if verified),
   cookie (if verified), and key unknowns.
D. Evidence-ranked shortlist. Rank by strength of evidence, not estimated
   earnings or the Market Discovery score.
E. Research gaps and next checks.

If the supplied evidence is insufficient for a candidate, do not fill the gap
with general knowledge. Mark it Not verified and explain what is missing.
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
        discovered_candidates, candidate_searches, candidate_search_results = (
            collect_candidate_specific_evidence(discovery, web_results)
        )
        official_site_evidence = collect_official_site_evidence(
            candidate_search_results
        )
        verification = run_verification_agent(
            discovery,
            candidate_searches,
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
            "version": "7.3.0",
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
            "discovered_candidates": discovered_candidates if "discovered_candidates" in locals() else [],
            "candidate_searches": candidate_searches if "candidate_searches" in locals() else [],
            "official_site_evidence": official_site_evidence if "official_site_evidence" in locals() else {},
            "source_results": web_results
        }

    except Exception as error:
        print(f"Unexpected workflow error in {current_agent}: {error}")
        return {
            "status": "partial",
            "project": "Affiliate Intelligence",
            "version": "7.3.0",
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
            "discovered_candidates": discovered_candidates if "discovered_candidates" in locals() else [],
            "candidate_searches": candidate_searches if "candidate_searches" in locals() else [],
            "official_site_evidence": official_site_evidence if "official_site_evidence" in locals() else {},
            "source_results": web_results
        }

    return {
        "status": "success",
        "project": "Affiliate Intelligence",
        "version": "7.3.0",
        "workflow": "Five-Agent Affiliate Intelligence System",
        "model": MODEL_NAME,
        "niche": request.niche,
        "search_results_found": len(web_results),
        "agents": completed,
        "source_results": web_results,
        "discovered_candidates": discovered_candidates,
        "candidate_searches": candidate_searches,
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
        "version": "7.3.0",
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
