import asyncio
import os
from unittest import result
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, AdaptiveCrawler
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy


async def crawl_snapshot(topic):
    # 1. Define the strategy: How deep and how many pages?
    # max_depth=2 follows links from the homepage to the next level
    strategy = BFSDeepCrawlStrategy(
        max_depth=1,
        include_external= True,
        max_pages=8
        ) 
    
    # 2. Configure the run to output clean Markdown
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        exclude_social_media_links=True, 
        word_count_threshold=200
    )

    async with AsyncWebCrawler() as crawler: 
        search_url = f"https://duckduckgo.com/html/?q={topic.replace(' ', '+')}"
        
        # 3. Use the new config with arun() instead of adaptive.digest()
        # This gives you direct control over the depth-based crawling
        results = await crawler.arun(url=search_url, config=config)

    # 4. Save the results
    for i, result in enumerate(results):
        if result.success and result.markdown:
            # result.markdown contains the clean text m.int needs
            content = result.markdown.raw_markdown 
            url = result.url
            
            filename = f"knowledge_base/research_{i}.txt"
            with open(filename, "w", encoding="utf-8") as f: 
                f.write(f"SOURCE: {url}\n\n")
                f.write(content)


def run_research_scout(topic):
    return asyncio.run(crawl_snapshot(topic))


