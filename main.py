
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
    description="AI-powered affiliate research, merchant discovery and verification platform",
    version="7.6.0"
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
    directory_names = {
        "the affiliates list", "saas affiliate", "affiliates list",
        "digistore24", "impact.com", "impact", "partnerstack",
        "awin", "shareasale", "cj affiliate", "commission junction",
        "clickbank", "rakuten advertising", "amazon associates",
        "class central", "lasso", "hostadvice"
    }
    generic_names = {
        "affiliate network", "affiliate directory", "affiliate marketplace",
        "affiliate program directory", "software affiliate programs",
        "physical products", "digital products", "online courses",
        "consumer technology", "technology products", "various brands",
        "affiliate programs", "affiliate platform"
    }

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
            normalized_name = candidate_name.lower().strip()
            if (
                not candidate_name
                or normalized_name in generic
                or normalized_name in generic_names
                or normalized_name in directory_names
            ):
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


def collect_market_page_evidence(search_results, max_pages: int = 14):
    """
    Inspect public search-result pages to extract underlying brands/products
    from directories, comparison articles and marketplace listings.
    These pages are discovery leads only, never proof of affiliate terms.
    """
    inspected = []
    seen = set()

    for item in search_results:
        url = item.get("url")
        if not url or url in seen:
            continue

        seen.add(url)
        page = inspect_public_page(url)

        compact_links = []
        for link in page.get("links", [])[:100]:
            href, label = link
            absolute = urljoin(page.get("final_url") or url, href)
            compact_links.append({
                "url": absolute,
                "label": label or ""
            })

        inspected.append({
            "search_title": item.get("title", ""),
            "search_url": url,
            "search_description": item.get("description", ""),
            "page_status": page.get("status", "unknown"),
            "page_title": page.get("title", ""),
            "page_description": page.get("description", ""),
            "page_text_excerpt": page.get("text_excerpt", "")[:4500],
            "outbound_and_internal_links": compact_links[:60],
            "fetch_reason": page.get("reason", "")
        })

        if len(inspected) >= max_pages:
            break

    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages_attempted": len(inspected),
        "pages": inspected
    }


# ==========================================
# AGENT 1 — MARKET DISCOVERY
# ==========================================

def run_discovery_agent(request, web_results, market_page_evidence=None):
    research_data = json.dumps(
        web_results,
        ensure_ascii=False,
        indent=2
    )
    page_evidence_data = json.dumps(
        market_page_evidence or {"pages": []},
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
- Aim for up to the requested maximum of DISTINCT MERCHANT BRANDS or
  specific products/services that a publisher could realistically promote.
- Search-page evidence includes directory and comparison pages. Treat those
  pages as sources to mine for the actual brands listed inside them.
- Do not count affiliate directories, affiliate networks, review blogs,
  comparison websites or affiliate software platforms as merchant opportunities.
- Seek a balanced mix across SaaS/software, physical consumer products,
  digital products, education/courses, technology and services when evidence
  supports them.
- Aim for at least 5 distinct categories and 10 concrete merchant candidates
  when supported by the supplied evidence. Never invent candidates to hit a quota.
"""
    else:
        category_strategy = """
FOCUSED DISCOVERY MODE:
- Respect the requested niche and prioritize relevant opportunities.
- You may include adjacent categories only when they are genuinely relevant.
- Do not force unrelated categories into the shortlist.
"""

    prompt = f"""
You are the Market Discovery Agent of Affiliate Intelligence, version 7.6.

REPORT DATE (UTC): {datetime.now(timezone.utc).date().isoformat()}

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

INSPECTED PUBLIC PAGE CONTENT AND LISTING LINKS (UNVERIFIED DISCOVERY EVIDENCE):
{page_evidence_data}

REQUIRED WORKFLOW:
1. Mine the inspected page text and links for the actual merchant brands,
   products and services named inside directories, roundups, comparison pages
   and category listings. Do not return the directory itself as a candidate.
2. For every extracted brand, retain the exact source page URL and, when
   available, its own official-domain URL. If the official domain is uncertain,
   mark it "Not identified" rather than guessing.
3. Extract distinct companies, brands, products and services from the search
   results and inspected page content.
4. Remove duplicates, directory brands, affiliate networks, publisher blogs,
   software used to manage affiliate programs, and irrelevant results.
5. Classify each candidate by product category and business model.
6. Assess whether the product solves a clear customer problem.
7. Identify likely target audiences and potential content angles.
8. Look for evidence that an affiliate, referral, partner or creator program
   may exist. A product being popular does not prove it has an affiliate program.
   Directory inclusion is third-party evidence only, never official confirmation.
   Do not state that a program is active or accepting applications in Discovery.
9. Assess recurring-revenue potential only where the business model supports it.
10. Compare candidates using the same scoring criteria.
11. Rank the strongest candidates and explain the trade-offs.

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
- Company / brand (must be a concrete merchant, not a directory/network)
- Product or service
- Category
- Official website URL (only if present in supplied evidence; never guess)
- Discovery source URL (exact supplied search/page URL)
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
B. Ranked comparison table of the strongest distinct merchant opportunities.
   Use one row per merchant. Include exact Discovery source URL and official
   website URL only when supplied evidence contains them.
C. Category breakdown (especially in broad discovery mode).
D. Directory extraction log: name each directory/listing page used and list
   the merchant brands extracted from it. Do not count the directory itself.
E. Top 3-10 concrete merchant candidates recommended for the Verification
   Agent, with reasons. Prefer up to 10 when evidence supports them.
F. Research gaps and limitations.

STRICT EVIDENCE RULES:
- Search results are leads, not proof of an active affiliate program.
- Never invent commission rates, cookie durations, prices, customer counts,
  conversion rates, program terms, approval requirements or earnings.
- Never claim a program is active unless the supplied evidence explicitly
  supports that status. The Verification Agent will check official sources.
- Mark unavailable information as "Not verified".
- Clearly separate observed evidence, analytical estimates and unknowns.
- Never use a directory, affiliate network, comparison blog or affiliate
  management platform as a merchant candidate merely because it appears in
  search results.
- If a directory lists individual merchants, extract those merchant names and
  cite the directory URL as discovery evidence. Their affiliate terms remain
  unverified until the Verification Agent checks official sources.
- If the supplied evidence supports fewer candidates than requested, return
  fewer and explain why. Quality and traceability matter more than quantity.
- The comparison table MUST use this exact first column format:
  | Rank | Company / Brand | Category | Product / Service | Official Website URL | Discovery Source URL | ...
  This allows the next agent to extract candidates reliably.

Return a structured, concise but sufficiently detailed professional report.
"""

    return generate_ai_response(prompt)



# ==========================================
# STRUCTURED EVIDENCE RECORDS
# ==========================================

VERIFICATION_STATUSES = {
    "official_affiliate_page_inspected_program_evidence_found",
    "official_website_inspected_affiliate_program_not_found",
    "third_party_evidence_only",
    "not_verified",
    "conflicting_evidence",
}

def build_candidate_evidence_inputs(candidates, candidate_searches, official_site_evidence):
    """Create deterministic, candidate-scoped evidence bundles from fetched sources."""
    pages = (official_site_evidence or {}).get("pages", [])
    search_by_name = {
        item.get("candidate", "").strip().lower(): item
        for item in (candidate_searches or [])
    }
    bundles = []

    for candidate in candidates:
        name = candidate.get("candidate", "")
        search_item = search_by_name.get(name.strip().lower(), {})
        search_results = search_item.get("results", [])
        candidate_pages = [
            page for page in pages
            if page.get("candidate", "").strip().lower() == name.strip().lower()
        ]

        evidence_items = []
        for result in search_results:
            evidence_items.append({
                "source_kind": "search_result_unverified",
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("description", "")[:1200],
                "page_inspected": False,
            })

        for page in candidate_pages:
            evidence_items.append({
                "source_kind": "fetched_public_page",
                "requested_url": page.get("requested_url", ""),
                "final_url": page.get("final_url", ""),
                "fetch_status": page.get("status", "unknown"),
                "page_title": page.get("title", ""),
                "page_description": page.get("description", ""),
                "text_excerpt": page.get("text_excerpt", "")[:5000],
                "fetch_reason": page.get("reason", ""),
                "page_inspected": page.get("status") == "inspected",
            })

        bundles.append({
            "candidate": name,
            "category": candidate.get("category", "Not stated"),
            "discovery_rank": candidate.get("rank"),
            "evidence_items": evidence_items,
        })

    return bundles


def _extract_json_payload(text_value):
    """Parse a JSON object/array from a model response, including fenced JSON."""
    cleaned = (text_value or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start_obj = cleaned.find("{")
    start_arr = cleaned.find("[")
    starts = [p for p in (start_obj, start_arr) if p >= 0]
    if not starts:
        raise ValueError("No JSON payload found in verification response.")
    start = min(starts)
    decoder = json.JSONDecoder()
    payload, _ = decoder.raw_decode(cleaned[start:])
    return payload


def _candidate_allowed_urls(bundle):
    allowed = set()
    for item in bundle.get("evidence_items", []):
        for key in ("url", "requested_url", "final_url"):
            value = item.get(key)
            if value:
                allowed.add(value)
    return allowed


def validate_structured_evidence_records(raw_records, evidence_bundles, checked_at):
    """
    Validate model output against actual candidate-scoped source URLs.
    Unsupported URLs/terms are downgraded rather than treated as verified.
    """
    if isinstance(raw_records, dict):
        raw_records = raw_records.get("records", [])
    if not isinstance(raw_records, list):
        raw_records = []

    raw_by_name = {
        str(item.get("candidate", "")).strip().lower(): item
        for item in raw_records if isinstance(item, dict)
    }

    validated = []
    for bundle in evidence_bundles:
        name = bundle["candidate"]
        raw = raw_by_name.get(name.strip().lower(), {})
        allowed_urls = _candidate_allowed_urls(bundle)
        inspected_urls = {
            item.get("final_url") or item.get("requested_url")
            for item in bundle.get("evidence_items", [])
            if item.get("source_kind") == "fetched_public_page"
            and item.get("page_inspected")
        }
        search_urls = {
            item.get("url") for item in bundle.get("evidence_items", [])
            if item.get("source_kind") == "search_result_unverified"
        }

        official_url = raw.get("official_website_url")
        affiliate_url = raw.get("official_affiliate_page_url")
        commission_source = raw.get("commission_source_url")
        cookie_source = raw.get("cookie_duration_source_url")

        # A model-supplied URL is retained only if this workflow actually found it.
        if official_url not in inspected_urls:
            official_url = None
        if affiliate_url not in inspected_urls:
            affiliate_url = None
        if commission_source not in inspected_urls:
            commission_source = None
        if cookie_source not in inspected_urls:
            cookie_source = None

        status = raw.get("verification_status")
        if status not in VERIFICATION_STATUSES:
            status = "not_verified"

        # Official-page inspection is not enough to verify a term: each term
        # must cite a candidate-specific page that this system actually fetched.
        commission = raw.get("commission")
        if not commission_source:
            commission = "Not verified"
        cookie_duration = raw.get("cookie_duration")
        if not cookie_source:
            cookie_duration = "Not verified"

        evidence_urls = []
        for value in raw.get("evidence_urls", []) if isinstance(raw.get("evidence_urls", []), list) else []:
            if value in allowed_urls and value not in evidence_urls:
                evidence_urls.append(value)

        # Conservative status corrections based on observable fetch coverage.
        if affiliate_url and status != "conflicting_evidence":
            status = "official_affiliate_page_inspected_program_evidence_found"
        elif inspected_urls and status not in {"conflicting_evidence"}:
            status = "official_website_inspected_affiliate_program_not_found"
        elif search_urls and status not in {"conflicting_evidence"}:
            status = "third_party_evidence_only"
        elif status != "conflicting_evidence":
            status = "not_verified"

        confidence = raw.get("confidence")
        if confidence not in {"high", "medium", "low", "unknown"}:
            confidence = "unknown"
        if status in {"third_party_evidence_only", "not_verified"}:
            confidence = "low" if search_urls else "unknown"

        validated.append({
            "candidate": name,
            "category": bundle.get("category", "Not stated"),
            "discovery_rank": bundle.get("discovery_rank"),
            "official_website_url": official_url,
            "official_affiliate_page_url": affiliate_url,
            "official_domain_confidence": confidence,
            "verification_status": status,
            "application_status": raw.get("application_status", "Unknown / not verified"),
            "commission": commission if commission is not None else "Not verified",
            "commission_source_url": commission_source,
            "cookie_duration": cookie_duration if cookie_duration is not None else "Not verified",
            "cookie_duration_source_url": cookie_source,
            "affiliate_program_evidence_found": bool(affiliate_url),
            "program_terms_verified": bool(commission_source or cookie_source),
            "evidence_urls": evidence_urls,
            "evidence_summary": str(raw.get("evidence_summary", ""))[:1200],
            "unknowns": raw.get("unknowns", []) if isinstance(raw.get("unknowns", []), list) else [],
            "last_checked_utc": checked_at,
            "confidence_note": (
                "Confidence applies to the evidence classification, not to earnings "
                "or the likelihood of approval."
            ),
            "source_coverage": {
                "search_results_found": len(search_urls),
                "public_pages_inspected": len(inspected_urls),
                "inspected_page_urls": sorted(inspected_urls),
            },
        })

    return validated


# ==========================================
# AGENT 2 — VERIFICATION
# ==========================================

def run_verification_agent(evidence_bundles, checked_at):
    """Return candidate-level structured evidence records and a readable report."""
    evidence_data = json.dumps(evidence_bundles, ensure_ascii=False, indent=2)

    prompt = f"""
You are the Verification Agent of Affiliate Intelligence, version 7.6.

DATE CHECKED (UTC): {checked_at}

TASK:
For every candidate in the supplied evidence bundles, return exactly one JSON
record. Use ONLY the evidence supplied for that same candidate.

Return valid JSON only, with this exact top-level shape:
{{
  "records": [
    {{
      "candidate": "exact candidate name",
      "official_website_url": "exact URL from supplied evidence or null",
      "official_affiliate_page_url": "exact URL of an inspected official affiliate page or null",
      "official_domain_confidence": "high|medium|low|unknown",
      "verification_status": "official_affiliate_page_inspected_program_evidence_found|official_website_inspected_affiliate_program_not_found|third_party_evidence_only|not_verified|conflicting_evidence",
      "application_status": "exactly stated status or Unknown / not verified",
      "commission": "exactly stated term or Not verified",
      "commission_source_url": "exact inspected source URL or null",
      "cookie_duration": "exactly stated term or Not verified",
      "cookie_duration_source_url": "exact inspected source URL or null",
      "evidence_urls": ["exact supplied URLs only"],
      "evidence_summary": "short factual explanation",
      "unknowns": ["unknown items"]
    }}
  ]
}}

EVIDENCE BUNDLES:
{evidence_data}

STRICT RULES:
- Include every candidate exactly once. Never add a candidate.
- Use exact source URLs from that candidate's evidence bundle only.
- A search result or third-party listing is not official verification.
- A fetched page is not automatically official. If domain ownership is uncertain,
  set confidence to low or unknown and do not call it an official affiliate page.
- An official website page that does not mention an affiliate program does not
  prove no program exists. Use the status "official_website_inspected_affiliate_program_not_found".
- Use "official_affiliate_page_inspected_program_evidence_found" only when an
  inspected page is demonstrably on the candidate's official domain AND itself
  contains affiliate/partner program evidence.
- A portal sign-in page, generic partner portal, or page that merely links to a
  program is not proof of commission terms or open applications.
- Commission and cookie duration must be exact text explicitly present on an
  inspected official page. If absent, use "Not verified" and null source URL.
- Never infer a commission, cookie, payout, geography, eligibility, approval
  requirement, program status or application availability.
- If a fetched page belongs to another company or a third-party network, do not
  attribute its terms to this candidate.
- Do not use general knowledge to fill gaps.
- If sources conflict, use "conflicting_evidence" and explain the conflict.
- Do not output markdown fences or commentary. JSON only.
"""

    raw_response = generate_ai_response(prompt)
    try:
        parsed = _extract_json_payload(raw_response)
    except Exception as exc:
        print(f"Verification JSON parse failed: {exc}")
        parsed = {"records": []}

    records = validate_structured_evidence_records(
        parsed, evidence_bundles, checked_at
    )

    report = {
        "date_checked_utc": checked_at,
        "candidate_count": len(records),
        "verification_status_counts": {
            status: sum(1 for item in records if item["verification_status"] == status)
            for status in sorted(VERIFICATION_STATUSES)
        },
        "records": records,
        "limitations": [
            "Public page inspection does not prove that a program is accepting applications.",
            "Terms are retained only when tied to an inspected candidate-specific source URL.",
            "A missing program reference on inspected pages does not prove that no program exists.",
        ],
    }
    return report, records




# ==========================================
# AGENT 3 — COMPETITION ANALYSIS
# ==========================================

def run_competition_agent(discovery_report, structured_evidence_records):

    evidence_data = json.dumps(structured_evidence_records, ensure_ascii=False, indent=2)
    prompt = f"""
You are the Competition Analysis Agent of Affiliate Intelligence, version 7.6.

STRUCTURED EVIDENCE RECORDS (single source of truth for candidate identity,
official URLs and affiliate verification status):
{evidence_data}

DISCOVERY CONTEXT (market leads only; not proof of affiliate terms):
{discovery_report}

Analyze competition and marketing opportunity for each listed candidate.

For every candidate:
- Keep the exact candidate name and verification_status from the structured record.
- Do not add, remove, merge or rename candidates.
- Do not claim verified commissions, cookie duration or program availability.
- If a field is "Not verified", preserve that uncertainty.
- Do not infer that a subscription product has recurring affiliate commissions.
- Label competition, SEO difficulty and channel assessments as qualitative estimates.
- Do not invent keyword volumes, traffic, market share or revenue.

Include:
1. Candidate and category.
2. Exact verification status.
3. Competition level: Low / Medium / High / Unknown.
4. Reason for qualitative competition assessment.
5. Potential audience and underserved segments.
6. SEO, YouTube, Instagram and LinkedIn opportunities where relevant.
7. Differentiation strategy and main risks.
8. Preliminary marketing attractiveness score from 1 to 10, explicitly an analytical estimate.
9. Evidence gaps that limit the assessment.

Return a concise structured report. Distinguish observed evidence from estimates.
"""
    return generate_ai_response(prompt)




# ==========================================
# AGENT 4 — FINANCIAL ANALYST
# ==========================================

def run_financial_agent(discovery_report, structured_evidence_records):

    evidence_data = json.dumps(structured_evidence_records, ensure_ascii=False, indent=2)
    prompt = f"""
You are the Financial Analyst Agent of Affiliate Intelligence, version 7.6.

STRUCTURED EVIDENCE RECORDS (sole authority for affiliate program status and terms):
{evidence_data}

DISCOVERY CONTEXT (unverified market leads; do not treat as confirmed terms):
{discovery_report}

For every candidate, assess financial potential and risks.

MANDATORY DATA RULES:
- Copy the exact verification_status from its structured evidence record.
- Use commission and cookie terms only when their corresponding source URL is present.
- If commission is "Not verified", do not calculate affiliate revenue for that candidate.
- Never infer recurring affiliate commissions from a subscription business model.
- Product prices, customer lifetime value, conversion rates, operating costs and
  earnings are NOT supplied as verified facts. Do not invent them.
- If useful, provide a clearly labeled hypothetical model with variables rather
  than invented dollar amounts or percentages. State that the user must supply
  actual traffic, conversion, commission and cost inputs.
- Never describe estimates as verified or say "verified via Verification Report"
  for any value that is not in the structured record.

For each candidate include:
1. Candidate and category.
2. Verification status.
3. Verified commission / cookie terms, or "Not verified".
4. Revenue model: known / possible / unknown, clearly distinguished.
5. Whether recurring affiliate revenue is explicitly verified (Yes / No).
6. Financial strengths and risks.
7. Cost categories to investigate; do not invent prices.
8. Key assumptions and missing inputs.
9. Financial attractiveness score from 1 to 10, explicitly an analytical estimate
   based on evidence quality and qualitative fit, not projected earnings.

Finish with a ranked comparison and a short list of data needed for a real forecast.
"""
    return generate_ai_response(prompt)




# ==========================================
# AGENT 5 — CHIEF STRATEGY
# ==========================================

def run_strategy_agent(
    discovery_report,
    structured_evidence_records,
    competition_report,
    financial_report
):
    evidence_data = json.dumps(structured_evidence_records, ensure_ascii=False, indent=2)
    prompt = f"""
You are the Chief Strategy Agent of Affiliate Intelligence, version 7.6.

STRUCTURED EVIDENCE RECORDS — SINGLE SOURCE OF TRUTH:
{evidence_data}

MARKET DISCOVERY REPORT (discovery leads only):
{discovery_report}

COMPETITION REPORT (qualitative estimates):
{competition_report}

FINANCIAL REPORT (must retain assumptions and unknowns):
{financial_report}

MISSION:
Recommend the strongest opportunities for an AI-powered affiliate media business.

NON-NEGOTIABLE DATA INTEGRITY:
- Use exactly the candidates in the structured evidence records.
- Copy each candidate's verification_status exactly.
- Do not describe a candidate as fully verified unless its status is
  "official_affiliate_page_inspected_program_evidence_found".
- An inspected affiliate page does not automatically mean terms or open
  applications are verified.
- Use commission/cookie terms only if present in the structured record with
  their corresponding source URL.
- Do not resolve contradictions by guessing.
- Never invent revenue, commission, traffic, pricing or earnings.
- Clearly separate verified facts, estimates and unknowns.

OUTPUT:
A. Executive summary of the market research and evidence quality.
B. Ranked shortlist with candidate, category, exact verification status,
   evidence quality, competition estimate, financial unknowns and rationale.
C. Top three recommendations for the NEXT RESEARCH STEP, not automatic promotion.
D. Lower-priority candidates and reasons.
E. Practical next actions to close evidence gaps.
F. Strategic conclusion.

Make clear that rankings/scores are preliminary analytical judgments, not measured
market performance. Return a professional structured report.
"""
    return generate_ai_response(prompt)




# ==========================================
# FIVE-AGENT ORCHESTRATOR
# ==========================================

def run_agent_workflow(request: ResearchRequest):
    """Run the five agents sequentially while preserving structured evidence."""
    web_results = collect_market_research(request.niche)

    if not web_results:
        raise HTTPException(
            status_code=503,
            detail="Web search returned no results. Please try again later."
        )

    report_generated_at_utc = datetime.now(timezone.utc).isoformat()
    completed = {}
    current_agent = "Market Discovery Agent"
    market_page_evidence = collect_market_page_evidence(web_results)
    discovered_candidates = []
    candidate_searches = []
    official_site_evidence = {}
    structured_evidence_records = []

    try:
        discovery = run_discovery_agent(request, web_results, market_page_evidence)
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

        evidence_bundles = build_candidate_evidence_inputs(
            discovered_candidates,
            candidate_searches,
            official_site_evidence
        )
        verification_checked_at = datetime.now(timezone.utc).isoformat()
        verification, structured_evidence_records = run_verification_agent(
            evidence_bundles,
            verification_checked_at
        )
        completed["verification"] = {
            "status": "completed",
            "report": verification,
            "structured_evidence_records": structured_evidence_records
        }

        current_agent = "Competition Analysis Agent"
        competition = run_competition_agent(
            discovery,
            structured_evidence_records
        )
        completed["competition"] = {
            "status": "completed",
            "report": competition
        }

        current_agent = "Financial Analyst Agent"
        financial = run_financial_agent(
            discovery,
            structured_evidence_records
        )
        completed["financial_analyst"] = {
            "status": "completed",
            "report": financial
        }

        current_agent = "Chief Strategy Agent"
        strategy = run_strategy_agent(
            discovery,
            structured_evidence_records,
            competition,
            financial
        )
        completed["chief_strategy"] = {
            "status": "completed",
            "report": strategy
        }

    except HTTPException as error:
        return {
            "status": "partial",
            "project": "Affiliate Intelligence",
            "version": "7.6.0",
            "report_generated_at_utc": report_generated_at_utc,
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
            "structured_evidence_records": structured_evidence_records,
            "market_page_evidence": market_page_evidence,
            "discovered_candidates": discovered_candidates,
            "candidate_searches": candidate_searches,
            "official_site_evidence": official_site_evidence,
            "source_results": web_results
        }

    except Exception as error:
        print(f"Unexpected workflow error in {current_agent}: {error}")
        return {
            "status": "partial",
            "project": "Affiliate Intelligence",
            "version": "7.6.0",
            "report_generated_at_utc": report_generated_at_utc,
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
            "structured_evidence_records": structured_evidence_records,
            "market_page_evidence": market_page_evidence,
            "discovered_candidates": discovered_candidates,
            "candidate_searches": candidate_searches,
            "official_site_evidence": official_site_evidence,
            "source_results": web_results
        }

    return {
        "status": "success",
        "project": "Affiliate Intelligence",
        "version": "7.6.0",
        "report_generated_at_utc": report_generated_at_utc,
        "workflow": "Five-Agent Affiliate Intelligence System",
        "model": MODEL_NAME,
        "niche": request.niche,
        "search_results_found": len(web_results),
        "agents": completed,
        "structured_evidence_records": structured_evidence_records,
        "market_page_evidence": market_page_evidence,
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
        "version": "7.6.0",
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

    market_page_evidence = collect_market_page_evidence(web_results)
    report = run_discovery_agent(
        request,
        web_results,
        market_page_evidence
    )

    return {
        "status": "success",
        "agent": "Market Discovery Agent",
        "model": MODEL_NAME,
        "search_results_found": len(web_results),
        "research_report": report,
        "market_page_evidence": market_page_evidence,
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
