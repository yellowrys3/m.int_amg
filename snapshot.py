"""
crawl4ai research collection for m.int's open-learning loop.

The main app runs generation from a local/closed environment, but the product
also needs a way to learn about current dental marketing and patient-education
topics. This module performs that open-web research step, saves the cleaned
results into `knowledge_base/`, and leaves indexing to `ingest.py`.
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, urlparse

# All crawl outputs are stored beside the practice pages so a single ingestion
# pass can embed both office facts and trend research into Chroma.
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
MIN_RESEARCH_WORDS = 180
MAX_SAVED_RESEARCH_FILES = 4
MAX_MARKDOWN_LINK_DENSITY = 0.12
BLOCKED_RESULT_DOMAINS = {
    "duckduckgo.com",
    "html.duckduckgo.com",
    "x.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
}
TOPIC_STOPWORDS = {
    "about",
    "best",
    "for",
    "from",
    "how",
    "latest",
    "near",
    "new",
    "patient",
    "patients",
    "question",
    "questions",
    "the",
    "tips",
    "what",
    "with",
}


def slugify_topic(topic):
    """
    Convert a free-form research topic into a safe filename component.

    The timestamp in `next_research_filename()` handles uniqueness; the slug
    makes the saved file easy to recognize later when browsing knowledge_base.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic.lower()).strip("_")
    return slug[:48] or "research"


def next_research_filename(topic, index):
    """
    Build a non-overwriting path for a crawl result.

    Older versions wrote `research_0.txt`, `research_1.txt`, etc., which meant a
    new Research Scout run could replace previous research. The timestamp/topic
    format keeps each crawl as an auditable snapshot.
    """
    KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify_topic(topic)
    return KNOWLEDGE_BASE_DIR / f"research_{timestamp}_{slug}_{index}.txt"


def topic_terms(topic):
    """
    Extract meaningful topic words used to judge crawl relevance.

    Search phrases often include filler words like "latest" or "tips." Removing
    those words keeps the relevance gate focused on the actual subject the user
    typed into Research Scout.
    """
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", topic.lower())
        if term not in TOPIC_STOPWORDS and len(term) > 2
    }


def normalized_result_url(url):
    """
    Return the real destination URL when DuckDuckGo wraps a search result.

    Saved source metadata should point to the page being researched, not the
    DuckDuckGo redirect URL that led to it.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "uddg" in query_params and query_params["uddg"]:
        return query_params["uddg"][0]

    return url


def is_blocked_domain(url):
    """
    Skip domains that usually produce search shells, login walls, or media pages.

    Those pages may contain many links but little reusable prose, which makes the
    knowledge base noisy and expensive to embed.
    """
    hostname = urlparse(url).hostname or ""
    hostname = hostname.removeprefix("www.")
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in BLOCKED_RESULT_DOMAINS)


def clean_research_markdown(markdown_text):
    """
    Strip obvious web chrome while preserving real article text.

    This happens before saving, so the knowledge_base folder contains useful
    prose rather than icon links, image placeholders, and long navigation lists.
    """
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", markdown_text)
    cleaned = re.sub(r"\[[^\]]{0,40}\]\([^)]*(login|signup|privacy|terms|cookie|pricing|contact)[^)]*\)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*\*\s+\[[^\]]+\]\([^)]*\)\s*$", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\b(log in|sign up|create account|cookie use|privacy policy|terms of service)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def research_quality(markdown_text, cleaned_text, topic):
    """
    Score whether a crawled page is worth saving as knowledge.

    A page must contain enough prose and avoid obvious link-shell behavior. Topic
    overlap is treated as a confidence boost rather than an absolute blocker,
    because good articles often use adjacent wording instead of the exact query.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", cleaned_text)
    word_count = len(words)
    if word_count < MIN_RESEARCH_WORDS:
        return False, f"too short ({word_count} words)"

    link_count = len(re.findall(r"\[[^\]]+\]\([^)]*\)", markdown_text))
    link_density = link_count / max(word_count, 1)
    if link_density > MAX_MARKDOWN_LINK_DENSITY:
        return False, f"too link-heavy ({link_density:.3f})"

    required_terms = topic_terms(topic)
    if required_terms:
        content_terms = {word.lower() for word in words}
        matches = required_terms & content_terms
        if not matches and word_count < 450:
            return False, "not enough topic overlap"

    return True, "usable"


async def crawl_snapshot(topic):
    """
    Crawl public web results for a dental research topic and save clean text.

    Args:
        topic: User-provided research phrase from the Streamlit sidebar.

    Returns:
        A list of saved file paths. The UI uses this count to confirm that new
        research was added before the user refreshes the Chroma index.

    Raises:
        RuntimeError: If crawl4ai is not installed. The import is kept inside
        the function so the rest of the app can run even when Research Scout's
        optional dependency has not been installed yet.
    """
    try:
        from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
        from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "crawl4ai is required for Research Scout. Install it with `pip install crawl4ai`."
        ) from exc

    # Follow the search-result page one level deep and cap page count so a
    # sidebar research run stays bounded and predictable.
    strategy = BFSDeepCrawlStrategy(
        max_depth=1,
        include_external= True,
        max_pages=8
        ) 
    
    # Ask crawl4ai for clean Markdown and skip tiny pages that usually contain
    # navigation, cookie notices, or thin search-result fragments.
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        exclude_social_media_links=True, 
        word_count_threshold=200
    )

    async with AsyncWebCrawler() as crawler: 
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(topic)}"
        
        # DuckDuckGo HTML is used as a lightweight discovery page. crawl4ai then
        # follows result links according to the BFS strategy above.
        results = await crawler.arun(url=search_url, config=config)

    saved_files = []
    saved_count = 0
    skipped_reasons = {}
    crawled_at = datetime.now().isoformat(timespec="seconds")

    # Save each successful result with source metadata at the top of the file.
    # These header lines become searchable text and also make the source clear
    # when a retrieved chunk is shown in the developer RAG test tab.
    for i, result in enumerate(results):
        if saved_count >= MAX_SAVED_RESEARCH_FILES:
            break

        if result.success and result.markdown:
            url = normalized_result_url(result.url)
            if is_blocked_domain(url):
                skipped_reasons["blocked domain"] = skipped_reasons.get("blocked domain", 0) + 1
                continue

            raw_content = result.markdown.raw_markdown
            content = clean_research_markdown(raw_content)
            is_usable, reason = research_quality(raw_content, content, topic)
            if not is_usable:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue
            
            filename = next_research_filename(topic, saved_count)
            with open(filename, "w", encoding="utf-8") as f:
                f.write("TYPE: trend_research\n")
                f.write(f"TOPIC: {topic}\n")
                f.write(f"SOURCE: {url}\n")
                f.write(f"CRAWLED_AT: {crawled_at}\n\n")
                f.write(content)
            saved_files.append(str(filename))
            saved_count += 1

    crawl_snapshot.last_skipped_reasons = skipped_reasons
    return saved_files


def run_research_scout(topic):
    """
    Synchronous wrapper used by Streamlit.

    Streamlit button handlers are regular synchronous code, while crawl4ai is
    async. This wrapper keeps that async detail out of `app.py`.
    """
    saved_files = asyncio.run(crawl_snapshot(topic))
    run_research_scout.last_skipped_reasons = getattr(crawl_snapshot, "last_skipped_reasons", {})
    return saved_files


run_research_scout.last_skipped_reasons = {}
crawl_snapshot.last_skipped_reasons = {}
