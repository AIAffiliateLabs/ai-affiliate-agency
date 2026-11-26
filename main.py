
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
# v8.3: version-consistent reports, semantic evidence enforcement, cross-agent consistency checks, and guarded strategy fallback.
# ==========================================

app = FastAPI(
    title="Affiliate Intelligence",
    description="AI-powered affiliate research, merchant discovery and verification platform",
    version="8.3.0"
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
# STRUCTURED EVIDENCE RECORDS — v7.8
# ==========================================

def _host(url):
    try:
        return (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception:
        return ""


def _candidate_key(name):
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _page_text(page):
    return " ".join([
        page.get("title", ""),
        page.get("description", ""),
        page.get("text_excerpt", "")
    ]).lower()


def _candidate_page_match(candidate_name, page):
    """Conservative association: candidate metadata must match, then host/title evidence."""
    if page.get("candidate", "").strip().lower() != candidate_name.strip().lower():
        return False
    key = _candidate_key(candidate_name)
    host = _candidate_key(_host(page.get("final_url") or page.get("requested_url", "")))
    title = _candidate_key(page.get("title", ""))
    # Require a recognizable brand token in either host or page title.
    return bool(key and (key in host or key in title))


def _commission_percentage_from_sentence(sentence):
    """Return a percentage only when grammar directly ties it to commission."""
    text = re.sub(r"\s+", " ", sentence or "").strip()
    percent = r"(\d{1,3}(?:\.\d{1,2})?\s*%)"
    patterns = [
        # "20% commission", "20% recurring commission", "20% affiliate commission"
        rf"{percent}\s*(?:(?:recurring|affiliate|referral)\s+)?commissions?\b",
        # "commission of 20%", "commission rate is 20%"
        rf"\bcommissions?\s*(?:rate\s*)?(?:of|is|:|at)\s*{percent}",
        # "earn 20% commission", "earn a 20% recurring commission"
        rf"\b(?:earn|earns|receive|receives|paid|pays?)\s+(?:a\s+)?(?:up\s+to\s+)?{percent}\s*(?:(?:recurring|affiliate|referral)\s+)?commissions?\b",
        # "earn commission of 20%"
        rf"\b(?:earn|earns|receive|receives|paid|pays?)\s+(?:a\s+)?(?:recurring\s+)?commissions?\s*(?:of|at)\s*{percent}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # The first capture in the first pattern is nested; use the final captured value.
            value = match.groups()[-1]
            return re.sub(r"\s+", "", value)
    return None


def _extract_explicit_terms(text):
    """Extract numeric terms only when the sentence semantically connects value and term."""
    commission = "Not verified"
    cookie = "Not verified"
    evidence = []
    normalized = re.sub(r"\s+", " ", text or "")
    # Split into sentence-like units to avoid borrowing unrelated page numbers.
    chunks = re.split(r"(?<=[.!?;])\s+|\n+", normalized)
    for chunk in chunks:
        sentence = chunk.strip()
        if not sentence or len(sentence) > 500:
            continue
        if commission == "Not verified":
            commission_value = _commission_percentage_from_sentence(sentence)
            if commission_value:
                commission = commission_value
                evidence.append({"term": "commission", "value": commission, "excerpt": sentence[:350]})
        if cookie == "Not verified" and re.search(r"(?i)\bcookie\b", sentence):
            m = re.search(r"(?i)\b(\d{1,4}\s*(?:days?|weeks?|months?))\b", sentence)
            if m:
                cookie = re.sub(r"\s+", " ", m.group(1)).strip()
                evidence.append({"term": "cookie_duration", "value": cookie, "excerpt": sentence[:350]})
        if commission != "Not verified" and cookie != "Not verified":
            break
    return commission, cookie, evidence


def _classify_program_type(pages):
    """Conservative classification; partner language alone is not an affiliate program."""
    text = " ".join(_page_text(page) for page in pages).lower()
    if any(x in text for x in ("affiliate program", "affiliate programme", "become an affiliate", "publisher program", "referral commission")):
        return "Classic affiliate/referral program — evidence language observed"
    if any(x in text for x in ("agency partner", "technology partner", "solution partner", "reseller partner", "integration partner", "partner directory")):
        return "B2B/agency/technology partnership — not confirmed as a publisher affiliate program"
    if any(x in text for x in ("partner program", "partner programme", "partners")):
        return "Partner program — type unclear; classic affiliate status not established"
    return "Unclear / no program type established"


def _candidate_host_match(candidate_name, page):
    """Require candidate identity in the fetched hostname; page titles alone are insufficient."""
    candidate_key = _candidate_key(candidate_name)
    host_key = _candidate_key(_host(page.get("final_url") or page.get("requested_url", "")))
    return bool(candidate_key and host_key and (candidate_key in host_key or host_key in candidate_key))


def _is_publisher_program_page(page):
    """Only explicit publisher-facing affiliate/referral language qualifies."""
    text = _page_text(page)
    url = (page.get("final_url") or page.get("requested_url", "")).lower()
    combined = f"{text} {url}"
    explicit = (
        "affiliate program", "affiliate programme", "become an affiliate",
        "join our affiliate", "affiliate application", "publisher program",
        "publisher programme", "referral program", "referral programme",
        "referral commission", "earn commission as an affiliate",
        "affiliate partners"
    )
    b2b_only = (
        "agency partner", "technology partner", "solution partner",
        "reseller partner", "integration partner", "implementation partner"
    )
    if any(term in combined for term in b2b_only) and not any(term in combined for term in explicit):
        return False
    return any(term in combined for term in explicit)


def _is_b2b_partner_page(page):
    combined = _page_text(page) + " " + (page.get("final_url") or page.get("requested_url", "")).lower()
    return any(term in combined for term in (
        "agency partner", "technology partner", "solution partner",
        "reseller partner", "integration partner", "implementation partner"
    ))


def build_structured_evidence_records(
    discovered_candidates,
    candidate_searches,
    official_site_evidence,
    market_sources=None
):
    """Create candidate-scoped records. Only candidate-host pages can support program terms."""
    pages = (official_site_evidence or {}).get("pages", [])
    search_by_candidate = {
        item.get("candidate", "").strip().lower(): item
        for item in (candidate_searches or [])
    }
    checked_at = (official_site_evidence or {}).get(
        "checked_at_utc", datetime.now(timezone.utc).isoformat()
    )
    records = []
    market_sources = market_sources or []

    for candidate in discovered_candidates or []:
        name = candidate.get("candidate", "")
        search_item = search_by_candidate.get(name.strip().lower(), {})
        search_results = search_item.get("results", [])
        tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) >= 3]
        related_market_sources = []
        for source in market_sources:
            title_desc = f"{source.get('title', '')} {source.get('description', '')}".lower()
            if tokens and any(re.search(r"\b" + re.escape(token) + r"\b", title_desc) for token in tokens):
                related_market_sources.append({
                    "title": source.get("title", ""), "url": source.get("url", ""),
                    "search_query": source.get("search_query", ""),
                    "evidence_type": "Candidate-mentioned market discovery lead — not program verification"
                })

        fetched = [p for p in pages if p.get("candidate", "").strip().lower() == name.strip().lower()
                   and p.get("status") == "inspected"]
        identity_pages = [p for p in fetched if _candidate_host_match(name, p)]
        nonmatching_pages = [p for p in fetched if p not in identity_pages]
        publisher_pages = [p for p in identity_pages if _is_publisher_program_page(p)]
        b2b_pages = [p for p in identity_pages if _is_b2b_partner_page(p)]

        discontinued_phrases = (
            "affiliate program has been discontinued", "affiliate programme has been discontinued",
            "affiliate program is discontinued", "affiliate program has ended",
            "affiliate program is no longer available", "we have discontinued our affiliate",
            "program has been discontinued", "programme has been discontinued"
        )
        discontinued_pages = [p for p in publisher_pages
                              if any(x in _page_text(p) for x in discontinued_phrases)]

        if discontinued_pages:
            status = "Program discontinued — do not promote"
            confidence = "Medium — explicit closure language on candidate-host page; ownership not independently authenticated"
        elif publisher_pages:
            status = "Candidate-host affiliate/referral page inspected — program evidence found"
            confidence = "Medium — candidate identity matched to hostname; legal ownership not independently authenticated"
        elif b2b_pages:
            status = "B2B/agency/technology partnership observed — publisher affiliate program not established"
            confidence = "Medium — candidate identity matched to hostname; partnership type is distinct from affiliate"
        elif identity_pages:
            status = "Candidate-host website inspected — affiliate program not found in inspected pages"
            confidence = "Medium — candidate identity matched to hostname; legal ownership not independently authenticated"
        elif search_results or nonmatching_pages:
            status = "Third-party evidence only"
            confidence = "Low — sources mention candidate but inspected host does not match candidate identity"
        else:
            status = "Not verified"
            confidence = "Unknown — no candidate-specific evidence retained"

        if status == "Program discontinued — do not promote":
            application_status = "Closed — explicit discontinuation evidence"
        elif publisher_pages:
            application_phrases = (
                "apply now", "apply to become an affiliate", "join our affiliate program",
                "sign up as an affiliate", "become an affiliate", "affiliate application"
            )
            application_status = (
                "Application link/invitation observed; acceptance not confirmed"
                if any(any(x in _page_text(p) for x in application_phrases) for p in publisher_pages)
                else "Program described; application status unclear"
            )
        else:
            application_status = "Not verified"

        program_evidence_pages = discontinued_pages or publisher_pages or b2b_pages
        evidence_urls = list(dict.fromkeys(
            p.get("final_url") or p.get("requested_url") for p in program_evidence_pages
            if p.get("final_url") or p.get("requested_url")
        ))
        official_url = (identity_pages[0].get("final_url") or identity_pages[0].get("requested_url")) if identity_pages else None

        commission, cookie_duration = "Not verified", "Not verified"
        term_evidence = []
        third_party_term_claims = []
        for page in nonmatching_pages:
            _, _, third_party_terms = _extract_explicit_terms(_page_text(page))
            for claim in third_party_terms:
                claim["evidence_url"] = page.get("final_url") or page.get("requested_url")
                claim["source_authority"] = "third-party or identity-mismatched host — never verified"
                third_party_term_claims.append(claim)
        for item in search_results:
            _, _, snippet_terms = _extract_explicit_terms(item.get("description", ""))
            for claim in snippet_terms:
                claim["evidence_url"] = item.get("url")
                claim["source_authority"] = "search snippet — lead only, never verified"
                third_party_term_claims.append(claim)
        for page in publisher_pages:
            page_text = _page_text(page)
            page_commission, page_cookie, extracted = _extract_explicit_terms(page_text)
            page_url = page.get("final_url") or page.get("requested_url")
            for item in extracted:
                item["evidence_url"] = page_url
                item["source_authority"] = "candidate-host page; domain ownership not independently authenticated"
                term_evidence.append(item)
            if commission == "Not verified" and page_commission != "Not verified":
                commission = page_commission
            if cookie_duration == "Not verified" and page_cookie != "Not verified":
                cookie_duration = page_cookie

        candidate_search_sources = [{
            "title": x.get("title", ""), "url": x.get("url", ""),
            "search_query": x.get("search_query", ""),
            "evidence_type": "Candidate-specific search lead — not official verification"
        } for x in search_results if x.get("url")]
        discovery_urls = list(dict.fromkeys(
            [x.get("url") for x in search_results if x.get("url")] +
            [x.get("url") for x in related_market_sources if x.get("url")]
        ))
        if search_results:
            traceability = "Candidate-specific sources retained"
        elif related_market_sources:
            traceability = "Contextual market sources retained; no candidate-specific search result"
        else:
            traceability = "No discovery source matched this candidate"

        if publisher_pages:
            program_type = "Classic publisher affiliate/referral program — explicit language observed"
        elif b2b_pages:
            program_type = "B2B/agency/technology partnership — not established as publisher affiliate"
        else:
            program_type = "Unclear / no publisher affiliate program established"

        records.append({
            "candidate": name,
            "category": candidate.get("category", "Not stated"),
            "discovery_rank": candidate.get("rank"),
            "official_url_candidate": official_url or "Not identified",
            "official_domain_confidence": "Not authenticated",
            "official_domain_match_confidence": confidence,
            "discovery_source_urls": discovery_urls,
            "candidate_specific_search_sources": candidate_search_sources,
            "related_market_discovery_sources": related_market_sources,
            "candidate_search_urls": list(dict.fromkeys(x.get("url") for x in search_results if x.get("url"))),
            "source_traceability_status": traceability,
            "inspected_page_urls": list(dict.fromkeys(
                p.get("final_url") or p.get("requested_url") for p in fetched
                if p.get("final_url") or p.get("requested_url")
            )),
            "affiliate_evidence_urls": evidence_urls,
            "verification_status": status,
            "program_status": status,
            "application_status": application_status,
            "program_type": program_type,
            "commission": commission,
            "cookie_duration": cookie_duration,
            "eligibility": "Not verified",
            "payout_threshold": "Not verified",
            "last_checked_utc": checked_at,
            "confidence": confidence,
            "term_evidence": term_evidence,
            "third_party_term_claims": third_party_term_claims,
            "unknowns": [
                "Legal ownership of the candidate domain is not independently authenticated.",
                "A program page does not prove applications are currently being accepted.",
                "Third-party commission and cookie claims are not accepted as verified terms.",
                "Eligibility, geography and payout details require direct confirmation unless explicitly shown on a qualifying candidate-host page."
            ]
        })

    return records




def _term_evidence_is_authoritative(record, term_name, value):
    """Require exact candidate-host evidence whose excerpt semantically supports the term."""
    official_host = _host(record.get("official_url_candidate", ""))
    if not official_host or not value or value == "Not verified":
        return False
    for evidence in (record.get("term_evidence") or []):
        if evidence.get("term") != term_name:
            continue
        if re.sub(r"\s+", "", str(evidence.get("value", ""))).lower() != re.sub(r"\s+", "", str(value)).lower():
            continue
        excerpt = str(evidence.get("excerpt", ""))
        evidence_url = evidence.get("evidence_url", "")
        evidence_host = _host(evidence_url)
        if term_name == "commission":
            parsed_value = _commission_percentage_from_sentence(excerpt)
            context_ok = bool(parsed_value and re.sub(r"\s+", "", parsed_value).lower() == re.sub(r"\s+", "", str(value)).lower())
        elif term_name == "cookie_duration":
            match = re.search(r"(?i)\bcookie\b", excerpt)
            duration = re.search(r"(?i)\b(\d{1,4}\s*(?:days?|weeks?|months?))\b", excerpt)
            parsed_value = re.sub(r"\s+", " ", duration.group(1)).strip() if duration else None
            context_ok = bool(match and parsed_value and parsed_value.lower() == str(value).lower())
        else:
            context_ok = False
        if context_ok and evidence_url and evidence_host == official_host:
            return True
    return False


def enforce_evidence_ledger(records):
    """Downgrade unsupported numeric terms before any downstream agent can consume them."""
    normalized = []
    enforcement_findings = []
    for original in (records or []):
        record = dict(original)
        for term_name in ("commission", "cookie_duration"):
            value = record.get(term_name, "Not verified")
            if value != "Not verified" and not _term_evidence_is_authoritative(record, term_name, value):
                enforcement_findings.append({
                    "candidate": record.get("candidate", ""),
                    "field": term_name,
                    "blocked_value": value,
                    "reason": "No focused exact-value excerpt from the candidate-host program page."
                })
                record[term_name] = "Not verified"
        normalized.append(record)
    return normalized, enforcement_findings


def _report_sentences(line):
    """Split a report line into sentence-like units for claim-level checks."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", str(line or "")) if part.strip()]


def _line_has_unsupported_term_claim(line, candidate):
    """Detect numeric and qualitative term claims that contradict the evidence ledger."""
    reasons = []
    commission = candidate.get("commission", "Not verified")
    cookie = candidate.get("cookie_duration", "Not verified")

    for sentence in _report_sentences(line):
        if commission == "Not verified" and _commission_percentage_from_sentence(sentence):
            reasons.append("commission is Not verified")
        if cookie == "Not verified":
            has_cookie_context = bool(re.search(r"(?i)\bcookie\b", sentence))
            has_duration = bool(re.search(r"(?i)\b\d{1,4}\s*(?:days?|weeks?|months?)\b", sentence))
            if has_cookie_context and has_duration:
                reasons.append("cookie duration is Not verified")
            elif has_cookie_context and re.search(
                r"(?i)\b(?:long|extended|generous|robust|attractive|lengthy|favorable|favourable)\s+cookie\b|\bcookie\s+(?:window|duration|period)\s+(?:is\s+)?(?:long|extended|generous|robust|attractive|favorable|favourable)\b",
                sentence
            ):
                reasons.append("qualitative cookie-duration claim is unsupported because cookie duration is Not verified")

    return list(dict.fromkeys(reasons))


def _normalize_report_version(report_text):
    """Replace stale self-reported platform versions in generated agent prose."""
    text = str(report_text or "")
    text = re.sub(r"(?i)(Affiliate Intelligence(?: System)?\s*,?\s*(?:version|v)\s*)8\.[0-2](?:\.0)?", r"\g<1>8.3.0", text)
    text = re.sub(r"(?i)(\bversion\s+)8\.[0-2](?:\.0)?", r"\g<1>8.3.0", text)
    text = re.sub(r"(?i)(\bv)8\.[0-2](?:\.0)?\b", r"\g<1>8.3.0", text)
    return text


def _sanitize_agent_report(report_text, records, agent_name):
    """Block report lines that state unsupported numeric or qualitative terms."""
    if not report_text:
        return report_text, []
    report_text = _normalize_report_version(report_text)
    lines = str(report_text).splitlines()
    findings = []
    by_name = {str(r.get("candidate", "")).lower(): r for r in records or []}
    names = sorted([n for n in by_name if n], key=len, reverse=True)
    output = []
    active_candidate = None

    for line in lines:
        for name in names:
            if re.search(r"(?i)(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", line):
                active_candidate = name
                break
        candidate = by_name.get(active_candidate) if active_candidate else None
        blocked_reasons = []

        if candidate:
            status = candidate.get("verification_status", "Not verified")
            if status in ("Third-party evidence only", "Not verified",
                          "Candidate-host website inspected — affiliate program not found in inspected pages",
                          "B2B/agency/technology partnership observed — publisher affiliate program not established"):
                if re.search(r"(?i)\b(?:verified shortlist|top recommendation|recommended opportunity|ranked (?:#?\d+|first|second|third)|ranked among verified|verified ranking)\b", line) and not re.search(
                    r"(?i)\b(?:not|never|cannot|can't|unverified|unknown|unclear|provisional|excluded|do not)\b", line
                ):
                    blocked_reasons.append("candidate is not verified and cannot be promoted as a verified recommendation")

            blocked_reasons.extend(_line_has_unsupported_term_claim(line, candidate))

        if blocked_reasons:
            findings.append({
                "candidate": candidate.get("candidate"),
                "agent": agent_name,
                "blocked_claim": line.strip()[:300],
                "reason": "; ".join(blocked_reasons)
            })
            output.append(
                f"[BLOCKED BY EVIDENCE ENFORCEMENT — {candidate.get('candidate')}: "
                f"{'; '.join(blocked_reasons)}] Claim withheld."
            )
        else:
            output.append(line)

    return "\n".join(output), findings


def build_enforcement_summary(records, ledger_findings, report_findings):
    return {
        "status": "blocked_claims" if ledger_findings or report_findings else "passed",
        "ledger_terms_downgraded": len(ledger_findings or []),
        "agent_claims_blocked": len(report_findings or []),
        "ledger_findings": ledger_findings or [],
        "blocked_agent_claims": report_findings or [],
        "rule": "Only exact numeric terms supported by focused candidate-host program-page excerpts remain verified. Commission percentages must be grammatically tied to commission; unrelated percentages such as conversion lifts are rejected. Unsupported numeric and qualitative term claims are replaced before downstream use."
    }


def build_enforced_strategy_report(records, enforcement_summary):
    """Deterministic fallback: never let a conflicting LLM strategy report reach the final output."""
    rows = []
    for record in records or []:
        name = record.get("candidate", "Unknown")
        status = record.get("verification_status", "Not verified")
        commission = record.get("commission", "Not verified")
        cookie = record.get("cookie_duration", "Not verified")
        program_type = record.get("program_type", "Unclear")
        rows.append(
            f"- **{name}** — Status: {status}; Program type: {program_type}; "
            f"Commission: {commission}; Cookie: {cookie}. "
            "Do not promote as a verified offer unless all required evidence is confirmed."
        )
    return (
        "# Evidence-Enforced Strategic Summary\n\n"
        "The AI-generated strategic ranking was withheld because the deterministic "
        "evidence controls detected unsupported claims. This report uses only the "
        "normalized Evidence Ledger.\n\n"
        "## Candidate status\n" + "\n".join(rows) +
        "\n\n## Enforcement outcome\n"
        f"- Ledger terms downgraded: {enforcement_summary.get('ledger_terms_downgraded', 0)}\n"
        f"- Unsupported agent claims blocked: {enforcement_summary.get('agent_claims_blocked', 0)}\n"
        "- No candidate is promoted by this fallback as a fully verified offer.\n"
        "- Next step: recover missing first-party sources and rerun verification."
    )


def _candidate_claim_lines(report_text, candidate_name):
    name_pattern = re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(candidate_name)}(?![A-Za-z0-9])")
    return [line.strip() for line in str(report_text or "").splitlines()
            if name_pattern.search(line) and line.strip()]


def build_agent_consistency_audit(records, agent_reports):
    """Validate evidence authority and consistency across all downstream agent reports."""
    findings = []
    records = records or []
    agent_reports = agent_reports or {}

    for record in records:
        name = record.get("candidate", "")
        status = record.get("verification_status", "Not verified")
        urls = record.get("discovery_source_urls", [])
        if not urls:
            findings.append({
                "candidate": name, "agent": "evidence_ledger", "ledger_status": status,
                "issue": "No discovery source URL retained; manual source recovery recommended",
                "severity": "source_gap"
            })

        for term_name, field_name in (("commission", "commission"), ("cookie_duration", "cookie_duration")):
            value = record.get(field_name, "Not verified")
            if not value or value == "Not verified":
                continue
            matches = [
                e for e in (record.get("term_evidence") or [])
                if e.get("term") == term_name
                and str(e.get("value", "")).replace(" ", "") == str(value).replace(" ", "")
            ]
            if not _term_evidence_is_authoritative(record, term_name, value):
                findings.append({
                    "candidate": name, "agent": "evidence_ledger", "ledger_status": status,
                    "issue": f"{field_name} is not supported by a focused excerpt on the candidate-host program page; value must be Not verified",
                    "severity": "unsupported_term", "reported_value": value,
                    "evidence_urls_reviewed": [e.get("evidence_url", "missing URL") for e in matches]
                })

        for agent_name, report in agent_reports.items():
            for line in _candidate_claim_lines(report, name):
                for reason in _line_has_unsupported_term_claim(line, record):
                    findings.append({
                        "candidate": name, "agent": agent_name, "ledger_status": status,
                        "issue": reason,
                        "claim_excerpt": line[:350],
                        "severity": "unsupported_agent_term"
                    })

                caveat_present = bool(re.search(
                    r"(?i)\b(?:not|never|cannot|can't|unable to|unverified|unknown|unclear|not confirmed|not established|do not|provisional|excluded|subject to confirmation)\b",
                    line
                ))

                if status == "Program discontinued — do not promote":
                    restricted = re.compile(r"(?i)\b(?:active affiliate program|active affiliate offer|currently accepting applications|open for applications|top recommendation|recommended for promotion|viable affiliate opportunity)\b")
                elif status in ("Third-party evidence only", "Not verified",
                                "Candidate-host website inspected — affiliate program not found in inspected pages",
                                "B2B/agency/technology partnership observed — publisher affiliate program not established"):
                    restricted = re.compile(r"(?i)\b(?:verified affiliate program|verified commission|verified cookie duration|currently accepting applications|open for applications|confirmed active affiliate program)\b")
                else:
                    restricted = re.compile(r"(?i)\b(?:currently accepting applications|open for applications|applications are open)\b")

                if restricted.search(line) and not caveat_present:
                    findings.append({
                        "candidate": name, "agent": agent_name, "ledger_status": status,
                        "issue": "Explicit candidate-level claim may conflict with evidence ledger",
                        "claim_excerpt": line[:350], "severity": "claim_conflict"
                    })

                if status in ("Third-party evidence only", "Not verified",
                              "Candidate-host website inspected — affiliate program not found in inspected pages",
                              "B2B/agency/technology partnership observed — publisher affiliate program not established"):
                    if re.search(r"(?i)\b(?:verified shortlist|top recommendation|recommended opportunity|ranked (?:#?\d+|first|second|third)|ranked among verified|verified ranking)\b", line) and not caveat_present:
                        findings.append({
                            "candidate": name, "agent": agent_name, "ledger_status": status,
                            "issue": "Unverified candidate promoted in a verified ranking or recommendation",
                            "claim_excerpt": line[:350], "severity": "claim_conflict"
                        })

    conflict_count = sum(
        1 for item in findings
        if item.get("severity") in ("claim_conflict", "unsupported_agent_term")
    )
    source_gap_count = sum(1 for item in findings if item.get("severity") == "source_gap")
    unsupported_term_count = sum(
        1 for item in findings if item.get("severity") == "unsupported_term"
    )
    return {
        "status": "review_required" if findings else "no_conflicts_detected_by_rules",
        "checks_run": len(records) * max(1, len(agent_reports)),
        "records_checked": len(records),
        "agent_reports_checked": len(agent_reports),
        "claim_conflicts": conflict_count,
        "source_gaps": source_gap_count,
        "unsupported_terms": unsupported_term_count,
        "findings": findings,
        "audit_method": (
            "Candidate-host identity and source-host checks; exact-value semantic evidence matching; "
            "sentence-level commission and cookie validation; unsupported qualitative cookie claims; "
            "candidate-level program-status and verified-ranking consistency checks."
        ),
        "note": (
            "Rule-based checks cannot authenticate legal domain ownership or understand every paraphrase. "
            "A clean audit is not proof that all claims are correct."
        )
    }


def build_report_metadata(agent_reports):
    """Report stale agent version labels without silently treating them as evidence issues."""
    stale = []
    version_pattern = re.compile(r"(?i)\b(?:Affiliate Intelligence(?: System)?\s*,?\s*(?:version|v)\s*|version\s+|v)8\.[0-2](?:\.0)?\b")
    for agent_name, report in (agent_reports or {}).items():
        if version_pattern.search(str(report or "")):
            stale.append(agent_name)
    return {
        "application_version": "8.3.0",
        "agent_reports_with_stale_version_labels": stale,
        "status": "passed" if not stale else "review_required",
        "version_policy": "All agent prompts and generated report version labels must match the running application version."
    }

def _records_json(records):
    return json.dumps(records or [], ensure_ascii=False, indent=2)


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
You are the Market Discovery Agent of Affiliate Intelligence, version 8.3.

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
# AGENT 2 — VERIFICATION
# ==========================================

def run_verification_agent(discovery_report, candidate_searches, official_site_evidence, structured_evidence_records):
    """Verify each candidate against its own targeted search and inspected pages."""
    research_data = json.dumps(candidate_searches, ensure_ascii=False, indent=2)
    inspected_data = json.dumps(official_site_evidence, ensure_ascii=False, indent=2)
    structured_data = _records_json(structured_evidence_records)

    prompt = f"""
You are the Verification Agent of Affiliate Intelligence, version 8.3.

V8 STRICT EVIDENCE RULES:
- The structured ledger is deterministic and authoritative; do not upgrade it.
- A candidate-host match is not legal ownership authentication.
- A page whose hostname does not match the candidate is third-party evidence, even if its title mentions the candidate.
- Generic product/home pages are not affiliate-program evidence. Require explicit publisher-facing affiliate/referral/publisher program language.
- Agency, technology, reseller, solution and integration partnerships are not classic publisher affiliate programs.
- Third-party commission/cookie claims must remain unverified; never copy them into official terms.
- A commission/cookie number is verified only if the ledger contains a focused excerpt and evidence URL from the same candidate-host program page.

REPORT DATE (UTC): {datetime.now(timezone.utc).date().isoformat()}

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

DETERMINISTIC STRUCTURED EVIDENCE RECORDS (SOURCE OF TRUTH):
{structured_data}

STRUCTURED RECORD RULES:
- Treat each structured record as the authoritative machine-generated evidence ledger.
- Repeat its candidate, URLs, status, commission and cookie fields exactly.
- Preserve candidate_specific_search_sources and related_market_discovery_sources as
  separate source classes. Contextual discovery sources are leads, not verification.
- For every material factual claim, provide the exact URL from the ledger or fetched pages.
- If no candidate-specific source URL exists, explicitly state "No candidate-specific
  discovery source retained" rather than inventing or borrowing another brand's source.
- You may explain evidence, but MUST NOT upgrade the record status or fill missing values.
- "Official URL candidate" is a candidate match, not independently authenticated ownership.
- Copy verification_status, program_status, application_status, official_domain_confidence,
  commission and cookie_duration exactly from the structured record.
- Never set application_status to "Open" merely because a page was inspected.
- If a record says terms are Not verified, keep them Not verified even if you know them from memory.

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

def run_competition_agent(discovery_report, verification_report, structured_evidence_records):

    structured_data = _records_json(structured_evidence_records)

    prompt = f"""
You are the Competition Analysis Agent of Affiliate Intelligence.

REPORT DATE (UTC): {datetime.now(timezone.utc).date().isoformat()}

DATA INTEGRITY CONTRACT:
- Treat the Verification Report as the sole authority for affiliate-program status and terms.
- Never upgrade a candidate's verification status.
- Do not repeat commission, cookie, payout, eligibility or application claims unless
  the Verification Report ties them to an inspected official page.
- Preserve exact uncertainty; if no matching candidate status is available, say Not verified.
- Keep estimates and qualitative competition judgments separate from facts.
- Cite source URLs from the ledger for candidate-specific factual claims; do not
  attach a general market source to a specific merchant unless the source names it.
- Reproduce the exact verification_status and application_status from the ledger.
- "Program discontinued — do not promote" must never appear in a viable shortlist,
  top recommendation, or active-program count.
- If status is Third-party evidence only, Not verified, or website inspected/program
  not found, do not describe the affiliate opportunity as verified or currently active.
- Do not give a numeric attractiveness score for discontinued programs; use N/A.
- For unverified program status, label the score "Provisional estimate — program unverified"
  and exclude the candidate from verified-program rankings.

Your mission is to evaluate the competitive landscape
for the discovered affiliate opportunities.

DISCOVERY REPORT:
{discovery_report}

VERIFICATION REPORT:
{verification_report}

STRUCTURED EVIDENCE LEDGER (DO NOT OVERRIDE):
{structured_data}

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
- Preliminary marketing attractiveness score from 1 to 10, or N/A if the
  affiliate program is discontinued. Mark unverified opportunities provisional
  and exclude them from the verified-program shortlist.

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

def run_financial_agent(discovery_report, verification_report, structured_evidence_records):

    structured_data = _records_json(structured_evidence_records)

    prompt = f"""
You are the Financial Analyst Agent of Affiliate Intelligence.

REPORT DATE (UTC): {datetime.now(timezone.utc).date().isoformat()}

DATA INTEGRITY CONTRACT:
- Treat the Verification Report as the sole authority for affiliate-program status and terms.
- Only report commission, cookie duration, payout threshold or program terms as verified
  if the Verification Report ties them to an inspected official page.
- Preserve statuses such as Third-party evidence only, Not verified, Conflicting evidence,
  or application status unclear.
- Never infer recurring commissions from a subscription product model.
- Label all modeled scenarios as hypothetical, not actual earnings.
- Cite exact ledger URLs for factual candidate claims. Contextual market discovery
  sources must not be presented as evidence of program terms or program availability.
- Use only ledger commission/cookie fields as verified terms; if Not verified, do not model numeric affiliate income for that candidate.
- Any operating-cost estimate must be explicitly labeled as a planning assumption, not sourced company data.
- Use the ledger's exact verification_status and application_status for every candidate.
- If program status is discontinued, financial attractiveness must be N/A and the
  candidate must be excluded from viable affiliate recommendations.
- If commission OR cookie terms are Not verified, do not calculate earnings, CLV,
  recurring revenue, or numeric payout scenarios for that candidate.
- A numeric term is usable only when its ledger term_evidence has a focused excerpt
  containing both the relevant keyword and the exact value, plus an evidence URL.
- Never treat a missing commission as 0%; missing means Not verified, not zero.
- Ignore third-party_term_claims for all numeric financial modelling.
- If program_type is B2B/agency/technology partnership, do not model it as a publisher affiliate offer.
- Do not assign a numeric financial score to a candidate without a verified program
  and verified commission basis. Use Not scored — insufficient verified evidence.
- B2B/agency/technology partnership is not automatically a publisher affiliate program.
- If program status is unverified or third-party-only, do not assign a numeric
  financial attractiveness score; use "Not scored — insufficient verified evidence".

Your responsibility is to evaluate the financial
potential and risks of each affiliate opportunity.

DISCOVERY REPORT:
{discovery_report}

VERIFICATION REPORT:
{verification_report}

STRUCTURED EVIDENCE LEDGER (AUTHORITATIVE TERM RECORDS):
{structured_data}

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
    financial_report,
    structured_evidence_records
):

    structured_data = _records_json(structured_evidence_records)

    prompt = f"""
You are the Chief Strategy Agent of Affiliate Intelligence.

REPORT DATE (UTC): {datetime.now(timezone.utc).date().isoformat()}

FINAL DATA INTEGRITY CONTRACT:
- Verification Report is the single source of truth for affiliate-program status and terms.
- Do not claim a program is active, accepting applications, or verified unless the
  Verification Report supports that exact candidate and statement.
- Preserve exact status; distinguish official evidence, third-party claims, estimates and unknowns.
- Cite the exact candidate-specific ledger URL beside every factual claim. Do not
  transfer evidence from another candidate or use contextual market leads as program proof.
- Disclose conflicts; do not silently choose or reconcile contradictory reports.
- Never call a candidate fully verified solely from a candidate/domain text match; domain ownership requires independent authentication.
- Distinguish candidate-domain match from authenticated official ownership.
- Never reuse another brand's commission, cookie, payout or eligibility terms.
- A third-party domain or generic product page cannot establish an affiliate program or its terms.
- Copy exact verification_status and application_status from the ledger.
- A discontinued program is disqualified: never rank it, recommend it, or describe
  it as a viable monetization opportunity.
- Do not call a candidate fully verified unless official domain ownership and
  current program terms have been independently authenticated; the current ledger
  does not independently authenticate domain ownership.
- Exclude unverified and third-party-only candidates from the verified shortlist.
- A B2B/agency/technology partnership is not a classic publisher affiliate program;
  classify it separately and do not recommend it as an affiliate offer without evidence.
- Never rank by commission or cookie values unless focused term_evidence supports them.
- Unknown commission is not zero and must never be converted to a zero-percent rate.
- Do not silently reconcile conflicts between reports; list them as unresolved.

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

STRUCTURED EVIDENCE LEDGER (FINAL AUTHORITY FOR TERMS):
{structured_data}

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

    report_generated_at_utc = datetime.now(timezone.utc).isoformat()
    completed = {}
    current_agent = "Market Discovery Agent"
    market_page_evidence = collect_market_page_evidence(web_results)
    discovered_candidates = []
    candidate_searches = []
    official_site_evidence = {}
    structured_evidence_records = []

    try:
        discovery = run_discovery_agent(
            request,
            web_results,
            market_page_evidence
        )
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
        structured_evidence_records = build_structured_evidence_records(
            discovered_candidates,
            candidate_searches,
            official_site_evidence,
            market_sources=web_results
        )
        structured_evidence_records, ledger_enforcement_findings = enforce_evidence_ledger(
            structured_evidence_records
        )
        discovery, discovery_blocked = _sanitize_agent_report(
            discovery, structured_evidence_records, "market_discovery"
        )
        completed["market_discovery"]["report"] = discovery
        verification = run_verification_agent(
            discovery,
            candidate_searches,
            official_site_evidence,
            structured_evidence_records
        )
        verification, verification_blocked = _sanitize_agent_report(
            verification, structured_evidence_records, "verification"
        )
        completed["verification"] = {
            "status": "completed",
            "report": verification
        }

        current_agent = "Competition Analysis Agent"
        competition = run_competition_agent(discovery, verification, structured_evidence_records)
        competition, competition_blocked = _sanitize_agent_report(
            competition, structured_evidence_records, "competition"
        )
        completed["competition"] = {
            "status": "completed",
            "report": competition
        }

        current_agent = "Financial Analyst Agent"
        financial = run_financial_agent(discovery, verification, structured_evidence_records)
        financial, financial_blocked = _sanitize_agent_report(
            financial, structured_evidence_records, "financial_analyst"
        )
        completed["financial_analyst"] = {
            "status": "completed",
            "report": financial
        }

        current_agent = "Chief Strategy Agent"
        strategy = run_strategy_agent(
            discovery,
            verification,
            competition,
            financial,
            structured_evidence_records
        )
        strategy, strategy_blocked = _sanitize_agent_report(
            strategy, structured_evidence_records, "chief_strategy"
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
            "version": "8.3.0",
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
            "market_page_evidence": market_page_evidence,
            "discovered_candidates": discovered_candidates if "discovered_candidates" in locals() else [],
            "candidate_searches": candidate_searches if "candidate_searches" in locals() else [],
            "official_site_evidence": official_site_evidence,
            "structured_evidence_records": structured_evidence_records,
            "evidence_enforcement": {"status": "partial_not_finalized", "note": "Workflow stopped before all agents completed."},
            "agent_consistency_audit": build_agent_consistency_audit(
                structured_evidence_records,
                {name: item.get("report", "") for name, item in completed.items()}
            ),
            "report_metadata": build_report_metadata(
                {name: item.get("report", "") for name, item in completed.items()}
            ),
            "source_results": web_results
        }

    except Exception as error:
        print(f"Unexpected workflow error in {current_agent}: {error}")
        return {
            "status": "partial",
            "project": "Affiliate Intelligence",
            "version": "8.3.0",
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
            "market_page_evidence": market_page_evidence,
            "discovered_candidates": discovered_candidates if "discovered_candidates" in locals() else [],
            "candidate_searches": candidate_searches if "candidate_searches" in locals() else [],
            "official_site_evidence": official_site_evidence,
            "structured_evidence_records": structured_evidence_records,
            "evidence_enforcement": {"status": "partial_not_finalized", "note": "Workflow stopped before all agents completed."},
            "agent_consistency_audit": build_agent_consistency_audit(
                structured_evidence_records,
                {name: item.get("report", "") for name, item in completed.items()}
            ),
            "report_metadata": build_report_metadata(
                {name: item.get("report", "") for name, item in completed.items()}
            ),
            "source_results": web_results
        }

    all_blocked_claims = (
        discovery_blocked + verification_blocked + competition_blocked +
        financial_blocked + strategy_blocked
    )
    enforcement_summary = build_enforcement_summary(
        structured_evidence_records, ledger_enforcement_findings, all_blocked_claims
    )
    agent_consistency_audit = build_agent_consistency_audit(
        structured_evidence_records,
        {
            "verification": verification,
            "competition": competition,
            "financial_analyst": financial,
            "chief_strategy": strategy
        }
    )
    critical_audit_findings = any(
        item.get("severity") in ("claim_conflict", "unsupported_agent_term", "unsupported_term")
        for item in agent_consistency_audit.get("findings", [])
    )
    if enforcement_summary["status"] == "blocked_claims" or critical_audit_findings:
        strategy = build_enforced_strategy_report(structured_evidence_records, enforcement_summary)
        completed["chief_strategy"]["report"] = strategy

    return {
        "status": "success",
        "project": "Affiliate Intelligence",
        "version": "8.3.0",
        "evidence_enforcement": enforcement_summary,
        "agent_consistency_audit": agent_consistency_audit,
        "report_metadata": build_report_metadata({
            name: item.get("report", "") for name, item in completed.items()
        }),
        "report_generated_at_utc": report_generated_at_utc,
        "workflow": "Five-Agent Affiliate Intelligence System",
        "model": MODEL_NAME,
        "niche": request.niche,
        "search_results_found": len(web_results),
        "agents": completed,
        "market_page_evidence": market_page_evidence,
        "source_results": web_results,
        "discovered_candidates": discovered_candidates,
        "candidate_searches": candidate_searches,
        "official_site_evidence": official_site_evidence,
        "structured_evidence_records": structured_evidence_records
    }


# ==========================================
# MAIN ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {
        "status": "online",
        "project": "Affiliate Intelligence",
        "version": "8.3.0",
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
