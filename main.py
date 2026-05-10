"""
M.int AMG V2
Created and Edited by: Kyle Zheng
Augmented by: OpenAI Codex with ChatGPT-5 Model & Ollama Gemma4:e4b Parameter Model

Core generation pipeline for the m.int dental marketing MVP.

This module owns the closed/local part of the application:
1. Load the configured dental office profile.
2. Retrieve relevant practice and trend context from the local Chroma store.
3. Ask the local Ollama model to generate an Instagram caption.
4. Ask the same model to turn that caption/topic into an engineered video prompt.

The Streamlit app should stay thin and call the orchestration functions here
instead of duplicating prompt, retrieval, or output-cleaning logic.
"""


### --- IMPORTS, ENVIRONMENT VARIABLES, PROFILES, and other GLOBALS [a] --- ###

## --  Import Modules [a1] -- ##  
from dotenv import load_dotenv
from datetime import datetime
import gc
import json
import os
import re
from pathlib import Path
from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langchain_community.vectorstores import Chroma                 # [Note to Self]: must update this module before support is pulled. 
from langchain_community.embeddings import HuggingFaceEmbeddings    


## -- Global Variables and Initialization [a2] -- ##

# Keep embeddings initialized once, but load Chroma lazily. Opening Chroma at
# import time can leave Streamlit holding a SQLite connection while the sidebar
# tries to rebuild `./chroma_db`, which can produce readonly-database errors.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vectorstore = None


# Initialize the local Ollama model with the specified parameters. 
llm = ChatOllama(model="gemma4:e4b",
            temperature=0.7,
)


# Regular expression pattern for tokenizing text, along with a set of common stopwords to exclude from tokenization. This is used in the lightweight local reranking function to help determine the relevance of retrieved context items based on term overlap with the query.
# (Cite: tokenize_for_ranking function) 
TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "before",
    "being", "but", "can", "care", "dental", "dentist", "for", "from",
    "has", "have", "how", "into", "its", "our", "that", "the", "their",
    "this", "through", "to", "with", "your",
}


## -- Environment Variables and API Keys [a3] -- ## 

# Load environment variables from .env file, including the keys for Langsmith and other necessary modules 
load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    print("✅ LangSmith API Key Loaded")
else:
    print("❌ LangSmith API Key Missing")


# Profile file for the business that the software is currently serving 
# [Note to Self]: This should be updated to support multiple profiles in the future, but for now it is hardcoded to Butterfly Dental Care. 
PROFILE_PATH = Path("business_profile.json")
DEFAULT_PROFILE = {
    "practice_name": "Butterfly Dental Care",
    "location": "San Jose, CA",
    "audience": "local dental patients",
    "tone": "warm, professional, and reassuring",
    "primary_cta": "Book an appointment",
    "website": "https://www.butterflydental.net/",
    "services": ["general dentistry", "cosmetic dentistry", "dental implants"],
    "content_guardrails": [
        "Do not promise guaranteed medical outcomes.",
        "Encourage patients to book a consultation for personalized advice.",
    ],
}


### --- CORE FUNCTIONS, CHAINS, and PARAMS [b] --- ###

## -- Business Profile Loading and Formatting [b1] -- ## 

@traceable(name="[b1] Load Business Profile")
def load_business_profile():
    """
    Load the dental office profile that personalizes every generated asset.

    The project is currently seeded for Butterfly Dental Care, but this function
    makes the pipeline portable to another dental office later. Values from
    `business_profile.json` override DEFAULT_PROFILE, while any missing fields
    safely fall back to the defaults above.
    """

    # Key Function Outline: 
    # 1) Check if the profile path exists. If it doesn't, return the default profile. 
    # 2) If the profile path exists, read the JSON file and merge it with the default profile, allowing for partial overrides.
    # 3) Return the merged profile for use in the generation pipeline.

    # In the instance that profile path could not be read, return default 
    if not PROFILE_PATH.exists():
        return DEFAULT_PROFILE

    # If the profile path exists, attempt to read it and merge with defaults. This allows for partial overrides without requiring every field to be specified in the JSON file.
    with PROFILE_PATH.open("r", encoding="utf-8") as profile_file:
        profile = json.load(profile_file)

    # Merge the loaded profile with the default profile, giving precedence to the loaded values.
    return {**DEFAULT_PROFILE, **profile}


@traceable(name="[b1] Format Business Profile")
def format_business_profile(profile):
    """
    Convert the structured business profile into prompt-ready text.

    LangChain prompt templates expect strings, not nested dictionaries. Keeping
    this formatting in one place also ensures the topic, caption, video prompt,
    and RAG test chains all receive the same practice context.
    """

    # Key Function Outline: 
    # 1) Format the Services List and Content Guardrails into readable strings because of the way they were originally formatted in business_profile
    # 2) Return all relevant profile information, structured in a way that is easy for the model to understand and use in generation. 

    # Format the services list into a comma-separated string for better readability in prompts 
    services = ", ".join(profile.get("services", []))

    # Format the content guardrails into a bulleted list for clearer presentation in prompts. Each guardrail is prefixed with a dash and a space for better readability.
    guardrails = "\n".join(f"- {item}" for item in profile.get("content_guardrails", []))
    
    # Return a formatted string that includes all relevant profile information, structured in a way that is easy for the model to understand and use in generation. 
    return (
        f"Practice: {profile.get('practice_name')}\n"
        f"Location: {profile.get('location')}\n"
        f"Audience: {profile.get('audience')}\n"
        f"Tone: {profile.get('tone')}\n"
        f"Primary CTA: {profile.get('primary_cta')}\n"
        f"Website: {profile.get('website')}\n"
        f"Services: {services}\n"
        f"Guardrails:\n{guardrails}"
    )


## -- Model Output Cleaning [b2] -- ##

@traceable(name="[b2] Clean Model Output")
def clean_model_output(raw_output):
    """
    Normalize model output before showing it in the UI.

    Some local reasoning models return hidden-thinking markers such as
    `</think>` or accidental tool-call wrappers. The marketing workspace should
    display only the usable caption or prompt, so this trims those artifacts and
    removes stale year hashtags that the prompt explicitly discourages.
    """

    # Removal of hidden-thinking markers and tool-call wrappers that some local reasoning models might include. Reduces artifacts seen by user 
    cleaned = raw_output.split("</think>")[-1]
    cleaned = cleaned.split("<tool_call>")[-1]
    cleaned = re.sub(r"#20\d{2}\b", "", cleaned)
    return cleaned.strip().strip('"')


## -- Local Generation Chains [b3] -- ## 

@traceable(name ="[b3] Topic Generation Chain")
def get_topic_generation_chain():
    """
    Build the topic-selection chain.

    The topic is generated from the business profile before RAG retrieval. This
    lets m.int choose topics that are useful for the office's audience and
    services instead of producing generic dental trivia.
    """

    # Key Functions: 
    # 1) Define a prompt template that instructs the model to suggest a specific, patient-friendly dental social media topic based on the business profile and retrieved context. 
    #       The prompt emphasizes topics tied to the practice's services, local audience, or current dental care questions, and it instructs the model to output only the topic without any explanation.
    # 2) Creates a chain with the prompt template, and specifies the llm used in the chain to be retrieved with subsequent pipelines 
    
    # Specifies the prompt fed to the model to suggest a topic (uses business profile and retrieved context).
    topic_prompt = ChatPromptTemplate.from_template(
        "You are a dental marketing strategist for this practice:\n"
        "{business_profile}\n\n"
        "Using what you know about the business and your research context and inherent knowledge,"
        "Suggest one specific, patient-friendly dental social media topic. "
        "Prefer topics tied to the practice's services, local audience, or current dental care questions. "
        "Output ONLY the topic, with no explanation."
    )

    # Specifies the chain that generates the topic, which includes the prompt, the model, and an output parser to ensure only the topic text is returned. 
    # [NOTE] The topic chain will be used in subsequent chains
    topic_chain = topic_prompt | llm | StrOutputParser()

    # Returns topic_chain 
    return topic_chain


@traceable(name ="[b3] Post Generation Chain")
def get_post_generation_chain():
    """
    Build the Instagram caption chain.

    This prompt combines the selected topic, the practice profile, and retrieved
    context. It also gives the model explicit dental-marketing safety rules:
    no diagnosis, no guaranteed outcomes, and no unsupported claims.
    """

    # Key Functions: 
    # 1) Define a prompt template that instructs the model to suggest a specific, patient-friendly dental social media post prompt based on the business profile, retrieved context, and selected topic.  
    # 2) Creates a chain with the post prompt and specifies the llm used in the chain to be retrieved with subsequent pipelines 

    # Specifies the prompt fed to the model to suggest a post prompt, baed on the topic, business profile, and contextual evidence
    post_prompt = ChatPromptTemplate.from_template(
        "You are writing social media copy for a small dental office.\n\n"
        "Business profile:\n{business_profile}\n\n"
        "Retrieved practice and research context:\n{context}\n\n"
        "Topic: {topic}\n\n"
        "Write one Instagram caption with this structure:\n"
        "1. Hook: one short question or bold statement.\n"
        "2. Value: 2-3 clear sentences using only safe, non-diagnostic dental language.\n"
        "3. Call to action: invite the reader to book or ask the practice a question.\n"
        "4. Hashtags: 3-5 relevant hashtags, with no year hashtags.\n\n"
        "Rules:\n"
        "- Treat the retrieved context as the source of truth for factual claims.\n"
        "- Use at least one concrete detail from a Practice Fact block when context is available.\n"
        "- Use Research blocks only when they directly support the topic.\n"
        "- Mention the practice or location only when it feels natural.\n"
        "- Do not promise guaranteed results or diagnose symptoms.\n"
        "- Do not invent services, credentials, prices, or promotions not supported by the context/profile.\n"
        "- Do not include citations or source labels in the caption itself.\n"
        "- Output only the final caption."
    )
    
    # Specifies the chain that generates the post caption, which includes the prompt, the model, and an output parser to ensure only the caption text is returned.
    post_chain = post_prompt | llm | StrOutputParser()
    return post_chain


@traceable(name = "[b3] Video Generation Chain") 
def get_video_generation_chain():
    """
    Build the engineered video-prompt chain.

    v1 does not generate video directly. Instead, it creates a structured prompt
    that can be pasted into an external video model or creative production tool.
    The generated prompt is tied to the caption so the post and video concept
    feel like one campaign asset rather than two unrelated outputs.
    """
    video_prompt = ChatPromptTemplate.from_template(
        "Create an engineered video-generation prompt for a short vertical dental marketing video.\n\n"
        "Business profile:\n{business_profile}\n\n"
        "Retrieved context:\n{context}\n\n"
        "Topic: {topic}\n"
        "Related caption:\n{caption}\n\n"
        "You are optimizing for Veo and LTX compatibility.\n"
        "Output exactly these sections in this exact order, each with its heading:\n"
        "1) Objective\n"
        "2) Format/Duration\n"
        "3) Scene Breakdown\n"
        "4) Camera/Lighting/Motion\n"
        "5) Audio/Mood\n"
        "6) On-screen Text\n"
        "7) Final CTA Frame\n"
        "8) Negative Constraints\n\n"
        "Rules:\n"
        "- Use 9:16 vertical format and 8-12 second runtime.\n"
        "- Keep language production-ready and concrete.\n"
        "- Ground details in the retrieved context and caption.\n"
        "- Keep patient-safe wording; no diagnosis language.\n"
        "- No guaranteed outcomes, no unsupported promotions, no graphic dental imagery.\n"
        "- Do not include citations, source labels, JSON, or extra sections.\n"
        "- Output only the final sectioned prompt."
    )

    video_chain = video_prompt | llm | StrOutputParser()
    return video_chain  


## -- VectorDB Initialization and Tokens Management [b4] -- ##

@traceable(name="[b4] Get Vector Store")
def get_vectorstore():
    """
    Return the active Chroma vectorstore, creating it only when retrieval runs.

    Lazy loading prevents the Streamlit app from opening the SQLite database just
    by importing `main.py`. That matters because the refresh workflow needs to
    delete and recreate the persisted Chroma files.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
        )
    return _vectorstore


@traceable(name="[b4] Reset Vector Store")
def reset_vectorstore():
    """
    Release the cached Chroma vectorstore before rebuilding the database.

    Chroma does not expose a simple universal close method through LangChain, so
    dropping the reference and forcing garbage collection is the safest local
    cleanup available before `ingest.py` removes `./chroma_db`.
    """
    global _vectorstore
    _vectorstore = None
    gc.collect()


@traceable(name="[b4] Tokenize for Ranking")
def tokenize_for_ranking(text):
    """
    Tokenize text for lightweight local reranking.

    This intentionally avoids adding a heavy reranker dependency. It gives the
    pipeline a deterministic baseline that works offline, while optional hybrid
    reranking can be added behind `external_rerank_context_items()`.
    """

    return {
        token for token in TOKEN_PATTERN.findall(text.lower()) # [Loops] through all tokens found in text (token for token) (.findall), converts them into lowercase (text.lower)
        if token not in STOPWORDS and len(token) > 2           # [Filters] out tokens that are in the STOPWORDS set (Cite: STOPWORDS defined above) and tokens that are 2 characters or shorter (len(token) > 2)
    }


## -- VectorDB Search Optimization and Database Query Management [b5] -- ##   

@traceable(name="[b5] Similarity Search with Fallback")
def similarity_search_with_fallback(query, knowledge_type, k=8):
    """
    Search Chroma by knowledge type, then fall back for old/unrefreshed stores.

    After the user refreshes the knowledge base, Chroma contains stable
    `knowledge_type` metadata. Before that refresh, an older local database may
    not have those fields, so this fallback keeps the app usable and surfaces
    lower-confidence results instead of returning an empty context package.
    """

    # Key Function: 
    # 1) Calls and loads Chroma VectorDB 
    # 2) First Attempt Search/Typed Docs: searches documents matching query AND specific knowledge type returning up to k results 
    # 3) [Check]: If first attempt returns result, return result. In the case it doesn't, trigger fallback. 
    # 4) Second Attempt Search: searches documents w/o knowledge type 

    # Calls the Chroma VectorDB, ensure DB is loaded and ready for search 
    store = get_vectorstore()

    # First attempt: search for documents matching the query and the specified knowledge type, returning up to k results. This is the preferred search method for refreshed databases with proper metadata.
    typed_docs = store.similarity_search(
        query,
        k=k,
        filter={"knowledge_type": knowledge_type},  # "knowledge_type" used as a filter to ensure that the search results are relevant to the specific type of knowledge being requested (e.g., "practice_profile" or "trend_research") 
    )

    # If the typed search returns results, then they are returned. If not, function triggers fallback method. 
    if typed_docs:
        return typed_docs

    # Second attempt (if needed): search for documents matching the query without filtering by knowledge type, returning up to k results. This fallback method is used for older databases that may not have the `knowledge_type` metadata.
    return store.similarity_search(query, k=max(3, k // 2))


@traceable(name="[b5] Retrieve Practice Context")
def retrieve_practice_context(topic, profile):
    """
    Retrieve office-specific facts for the selected marketing topic.

    Practice context should answer questions like who the office is, where it is,
    what services it offers, and which CTA is appropriate.
    """

    # Key Function Outline:
    # 1) Initializes service_terms from business profile 
    # 2) Constructs a search query that combines practice name, location, service_terms, primary_cta, and topic
    # 3) Returns the results of the similarity search with fallback using the constructed query 

    # Initializes service_terms from the business profile (Cite: default_profile and load_business_profile function) to ensure that the search query includes relevant services offered by the dental practice.
    service_terms = " ".join(profile.get("services", []))
    
    # Constructs a search query that combines both practice name, location, the previous-built service_terms above, the primary cta, and the topic generated (Cite: topic_generation_chain)
    query = (
        f"{profile.get('practice_name')} {profile.get('location')} "
        f"{profile.get('primary_cta')} {service_terms} {topic}"
    )

    # Returns the results of the similarity search with fallback, specifying "practice_profile" as the knowledge type to prioritize office-specific facts in the search results. The function will return up to 8 relevant documents from the Chroma vectorstore that match the query and knowledge type.
    return similarity_search_with_fallback(query, "practice_profile", k=8)


@traceable(name="[b5] Retrieve Research Context")
def retrieve_research_context(topic):
    """
    Retrieve trend or educational context for the selected marketing topic.

    Research context is intentionally separate from practice context so the
    caption can combine office facts with broader patient-education material
    without letting generic web research drown out the local practice details.
    """

    ### PLACEHOLDER ###
    return similarity_search_with_fallback(topic, "trend_research", k=8)


@traceable(name="[b5] External Rerank Context Items")
def external_validation_report(caption, video_prompt, context_items, profile):
    """
    Optional hybrid validation hook.

    A future version can call a stronger reviewer model or external policy
    service here. Returning None keeps the current MVP on the deterministic
    local checker below.
    """

    ### PLACEHOLDER ### 
    if not os.getenv("MINT_EXTERNAL_VALIDATOR"):
        return None

    return None


@traceable(name="[b5] Build Context Item")
def build_context_item(doc, query_terms, profile, preferred_type):
    """
    Convert a LangChain document into a scored context item for generation/UI.

    The score combines semantic retrieval order from Chroma with transparent
    local signals: query-term overlap, source type, service/profile matches,
    and research recency.
    """

    # Key Functions Outline: 
    # 1) Initializes/Extract metadata and content from the document, and calculate the overlap between query terms and content terms to determine relevance.
    # 2) Add small boost if content includes any of the services offered by the practice, as this can indicate higher relevance to the practice's offerings and the marketing topic. 
    # 3) Returns dictionary representing the context item 


    # Initialization of different aspects of data 
    metadata = dict(doc.metadata)
    content = doc.page_content.strip()
    content_terms = tokenize_for_ranking(content)
    overlap = len(query_terms & content_terms)
    overlap_score = overlap / max(len(query_terms), 1)
    knowledge_type = metadata.get("knowledge_type", preferred_type)
    type_score = 0.25 if knowledge_type == preferred_type else 0.0
    service_text = " ".join(profile.get("services", [])).lower()
    service_matches = sum(1 for service in profile.get("services", []) if service.lower() in content.lower())
    profile_score = 0.15 if profile.get("practice_name", "").lower() in content.lower() else 0.0
    service_score = min(service_matches * 0.05, 0.2)
    recency_score = parse_recency_score(metadata.get("crawled_at", "")) * 0.1
    score = overlap_score + type_score + profile_score + service_score + recency_score

    # Add a small boost if the content includes any of the services offered by the practice, as this can indicate higher relevance to the practice's offerings and the marketing topic. The boost is capped to prevent it from dominating the overall score, ensuring that other relevance signals still play a significant role in the ranking.
    if service_text and any(term in content.lower() for term in service_text.split()):
        score += 0.05
    
    # Return a dictionary representing the context item, including the content, metadata, knowledge type, calculated score, and source information for use in generation and display in the UI. 
    return {
        "content": content,
        "metadata": metadata,
        "knowledge_type": knowledge_type,
        "score": round(score, 4),
        "source_title": metadata.get("source_title") or Path(metadata.get("source_path", "")).stem,
        "source_path": metadata.get("source_path") or metadata.get("source", ""),
        "source_url": metadata.get("source_url", ""),
        "topic": metadata.get("topic", ""),
        "crawled_at": metadata.get("crawled_at", ""),
        "chunk_index": metadata.get("chunk_index", ""),
    }


@traceable(name="[b5] Rerank Context Items")
def rerank_context_items(query, docs, profile, preferred_type):
    """
    Locally rerank retrieved documents and suppress duplicates/source crowding.

    This is the default reranker for the local-first MVP. It is intentionally
    explainable: the returned items include score and metadata so the developer
    RAG tab can show why a chunk reached the generation context.
    """
    
    
    query_terms = tokenize_for_ranking(query)
    scored_items = [
        build_context_item(doc, query_terms, profile, preferred_type)
        for doc in docs
        if doc.page_content and doc.page_content.strip()
    ]

    external_items = external_rerank_context_items(query, scored_items)
    if external_items:
        return external_items

    scored_items.sort(key=lambda item: item["score"], reverse=True)
    selected_items = []
    seen_fingerprints = set()
    source_counts = {}

    for item in scored_items:
        fingerprint = re.sub(r"\W+", "", item["content"].lower())[:260]
        source_key = item["source_url"] or item["source_path"] or item["source_title"]

        if fingerprint in seen_fingerprints:
            continue
        if source_counts.get(source_key, 0) >= 2:
            continue

        seen_fingerprints.add(fingerprint)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        selected_items.append(item)

    return selected_items


@traceable(name="[b5] Format Context Package")
def format_context_package(items, max_items=6, max_chars_per_item=900):
    """
    Format ranked context into a compact, labeled prompt package.

    Labels make it harder for the model to blur practice facts and trend
    research together, and the character cap prevents a few large chunks from
    crowding out the rest of the evidence.
    """
    context_blocks = []
    for index, item in enumerate(items[:max_items], start=1):
        label = "Practice Fact" if item["knowledge_type"] == "practice_profile" else "Research"
        source = item["source_title"] or item["source_url"] or item["source_path"] or "Unknown source"
        excerpt = item["content"][:max_chars_per_item].strip()
        context_blocks.append(
            f"[{index}] {label} | score={item['score']} | source={source}\n{excerpt}"
        )

    return "\n\n".join(context_blocks)


@traceable(name="[b5] Retrieve Context")
def retrieve_context(query):
    """
    Backward-compatible retrieval wrapper used by older callers.

    New generation uses `retrieve_ranked_context()`, but this keeps simple RAG
    testing or notebooks from breaking if they still expect `(context, sources)`.
    """
    profile = load_business_profile()
    items = retrieve_ranked_context(query, profile)
    return format_context_package(items), [item["metadata"] for item in items]


@traceable(name="[b5] Retrieve Ranked Context")
def retrieve_ranked_context(topic, profile):
    """
    Retrieve and rerank practice and research context as separate evidence sets.

    The returned list alternates strong practice facts with relevant research so
    generation receives both office-specific grounding and broader topic support.
    """
    practice_docs = retrieve_practice_context(topic, profile)
    research_docs = retrieve_research_context(topic)
    practice_query = f"{profile.get('practice_name')} {profile.get('location')} {topic}"
    research_query = topic

    practice_items = rerank_context_items(practice_query, practice_docs, profile, "practice_profile")[:4]
    research_items = rerank_context_items(research_query, research_docs, profile, "trend_research")[:4]

    merged_items = []
    for pair_index in range(max(len(practice_items), len(research_items))):
        if pair_index < len(practice_items):
            merged_items.append(practice_items[pair_index])
        if pair_index < len(research_items):
            merged_items.append(research_items[pair_index])

    return merged_items


@traceable(name="[b5] Retrieve Prompt Playbook Context")
def retrieve_prompt_playbook_context(topic, profile):
    """
    Retrieve prompt-engineering guidance for Veo/LTX-style video prompting.

    This third lane complements practice and trend context with model-agnostic
    production wording patterns sourced from curated prompt playbook docs.
    """
    service_terms = " ".join(profile.get("services", []))
    query = (
        f"veo ltx video prompt guide dental marketing "
        f"{service_terms} {topic} 9:16 scene camera lighting audio cta"
    )
    docs = similarity_search_with_fallback(query, "prompt_playbook", k=6)
    return rerank_context_items(query, docs, profile, "prompt_playbook")[:3]


@traceable(name="[b5] Retrieve Ranked Video Context")
def retrieve_ranked_video_context(topic, profile):
    """
    Retrieve mixed evidence for video prompt generation.

    This includes practice facts, trend research, and prompt-playbook guidance.
    """
    base_items = retrieve_ranked_context(topic, profile)
    playbook_items = retrieve_prompt_playbook_context(topic, profile)
    if not playbook_items:
        return base_items

    merged_items = []
    playbook_inserted = False
    for item in base_items:
        merged_items.append(item)
        if not playbook_inserted and len(merged_items) >= 2:
            merged_items.extend(playbook_items)
            playbook_inserted = True

    if not playbook_inserted:
        merged_items.extend(playbook_items)

    return merged_items


@traceable(name="[b5] Context-Term Overlap")
def context_term_overlap(output_text, context_items):
    """
    Count meaningful overlap between generated output and retrieved evidence.

    This is a lightweight grounding signal. It does not prove factual support,
    but it catches the failure mode where useful retrieved context exists and
    the model still writes a fully generic caption or video prompt.
    """
    output_terms = tokenize_for_ranking(output_text)
    context_terms = set()
    for item in context_items[:6]:
        context_terms.update(tokenize_for_ranking(item["content"]))
        context_terms.update(tokenize_for_ranking(item.get("source_title", "")))

    return len(output_terms & context_terms)


### --- Crawl (Research Scout) and Reranking Helpers [b6] --- ### 

@traceable(name="[b6] Parse Recency Score")
def parse_recency_score(crawled_at):
    """
    Convert Research Scout timestamps into a small reranking boost.

    Freshness matters for trend research but should never dominate relevance, so
    this score is deliberately modest and safely falls back to zero.
    """

    # Key Function Outline: 
    # 1) [Check] if crawled_at timestamp is present, if not, return base value of 0 
    # 2) If timestamp is present, attempt to parse. If parsing fails, return base value of 0
    # 3) Intialize age_days (calculates age of crawled context and prevents negative age due to error)

    # If info was not crawled, return base value of 0 
    if not crawled_at:
        return 0.0

    # If timestamp is present, attempt to parse. If parsing fails, return base value of 0 
    try:
        crawled_date = datetime.fromisoformat(crawled_at)
    except ValueError:
        return 0.0

    # Initialize age_days, calculates age of crawled context and prevents negative age due to error
    age_days = max((datetime.now() - crawled_date).days, 0)

    # Convert age in days to a recency score between 0 and 0.1, where more recent items receive a higher boost.
    return max(0.0, 1.0 - (age_days / 365))


@traceable(name="[b6] External Rerank Context Items")
def external_rerank_context_items(query, context_items):
    """
    Optional hybrid reranking hook.

    The default MVP path is fully local. This placeholder keeps the call site
    ready for an external reranker later without requiring API keys or changing
    the generation pipeline today.
    """
    if not os.getenv("MINT_EXTERNAL_RERANKER"):
        return None

    return None


## -- [Check] Grounding/Validation Helpers for RAG Pipeline [b7] -- ## 

@traceable(name="[b7] Build Answer Grounding Report")
def build_answer_grounding_report(answer, context_items):
    """
    Check whether a Developer RAG answer cites retrieved context blocks.

    Developer RAG is a debugging surface, so visible citations are useful there:
    they make it clear whether the answer came from retrieved evidence or from
    unsupported model memory.
    """

    # Key Function Outline: 
    # 1) If there was no context provided, return a report indicating that no supporting context was retrieved, along with an empty list of cited context IDs. 
    # 2) Initialize max_context_id based on the number of context items, ensuring it does not exceed 6.
    # 3) Initialize cited_ids by extracting all numbers enclosed in square brackets from the answer text using a regular expression. The function filters these numbers to include only those that are valid context block IDs (between 1 and max_context_id) and converts them to integers. The resulting list of cited IDs is sorted for consistency in the report.
    # 4) If cited_ids is not empty, return a report indicating that the answer

    # If there was no context provided, return a report indicating that no supporting context was retrieved, along with an empty list of cited context IDs. 
    if not context_items:
        return {
            "status": "no_context",
            "message": "No supporting context was retrieved.",
            "cited_context_ids": [],
        }

    # Initialize max_context_id based on the number of context items, ensuring it does not exceed 6. This is important because the answer may only cite context blocks that are included in the generation prompt, which is limited to the top 6 context items.
    max_context_id = min(len(context_items), 6)

    # Initialize cited_ids by extracting all numbers enclosed in square brackets from the answer text using a regular expression. The function filters these numbers to include only those that are valid context block IDs (between 1 and max_context_id) and converts them to integers. The resulting list of cited IDs is sorted for consistency in the report.
    cited_ids = sorted({
        int(match)
        for match in re.findall(r"\[(\d+)\]", answer)
        if 1 <= int(match) <= max_context_id
    })

    # If cited_ids is not empty, return a report indicating that the answer cited retrieved context blocks, along with a message listing the cited context block IDs and the list of cited context IDs. If cited_ids is empty, return a report indicating that the retrieved context existed but the answer did not cite it, along with an empty list of cited context IDs.
    if cited_ids:
        return {
            "status": "grounded",
            "message": f"Answer cited retrieved context block(s): {', '.join(f'[{item_id}]' for item_id in cited_ids)}.",
            "cited_context_ids": cited_ids,
        }

    # Otherwise, if cited_ids is empty, return a report indicating that the retrieved context existed but the answer did not cite it, along with an empty list of cited context IDs.
    return {
        "status": "needs_review",
        "message": "Retrieved context existed, but the answer did not cite it.",
        "cited_context_ids": [],
    }


@traceable(name="[b7] Build Validation Report")
def build_validation_report(caption, video_prompt, context_items, profile):
    """
    Review generated outputs before they are shown as final marketing assets.

    This deterministic checker catches common v1 risks without requiring another
    model call: unsupported certainty, diagnosis language, stale hashtags,
    missing CTA, weak grounding, and incomplete video prompt structure.
    """

    # Initiate warnings and checks lists to collect any issues found during validation and to keep track of the various checks performed on the generated caption and video prompt.
    warnings = []
    checks = []

    # Initiate combined_output by concatenating both the caption and the video prompt
    combined_output = f"{caption}\n\n{video_prompt}".lower()

    ### External Report Placeholder ###
    external_report = external_validation_report(caption, video_prompt, context_items, profile)
    if external_report:
        return external_report

    # Helper Function (add_check): to add a check result to the checks list and optionally add a warning if the check did not pass. 
    def add_check(name, passed, note):
        checks.append({"name": name, "passed": passed, "note": note})
        if not passed:
            warnings.append(note)

    # Helper Globals for add_check:
    
    # Risk check helper globals  
    risky_terms = ["guaranteed", "cure", "diagnose", "you have", "permanent results", "pain-free"]
    risky_hits = [term for term in risky_terms if term in combined_output]

    # CTA check helper globals 
    cta_text = profile.get("primary_cta", "").lower()
    cta_terms = {"book", "appointment", "schedule", "consultation", "call", "ask"}

    # Practice relevance check helper globals (Cite: practice relevance add_check)
    practice_markers = [
        profile.get("practice_name", "").lower(),
        profile.get("location", "").lower(),
        *[service.lower() for service in profile.get("services", [])],
    ]
    grounding_hits = [marker for marker in practice_markers if marker and marker in combined_output]

    # Evidence overlap check helper globals (Cite: evidence overlap add_check)
    evidence_overlap = context_term_overlap(combined_output, context_items)

    # Video and video prompt completeness check helper globals (Cite: video completeness add_check)
    video_requirements = {
        "vertical format": ["9:16", "vertical"],
        "runtime": ["second", "runtime", "8-12", "8 to 12"],
        "scene direction": ["scene", "shot", "close-up", "camera"],
        "audio or mood": ["audio", "music", "sound", "mood"],
        "text overlays": ["text", "overlay", "on-screen"],
    }
    missing_video_parts = [
        name for name, terms in video_requirements.items()
        if not any(term in video_prompt.lower() for term in terms)
    ]
    required_schema_sections = [
        "objective",
        "format/duration",
        "scene breakdown",
        "camera/lighting/motion",
        "audio/mood",
        "on-screen text",
        "final cta frame",
        "negative constraints",
    ]
    normalized_video_prompt = video_prompt.lower()
    missing_schema_sections = [
        section for section in required_schema_sections
        if section not in normalized_video_prompt
    ]

    # Unsafe terms and model-safe wording check helper globals (Cite: model-safe wording add_check)
    unsafe_model_terms = ["diagnose", "guaranteed", "guarantee", "cure", "before/after"]
    unsafe_model_hits = [term for term in unsafe_model_terms if term in normalized_video_prompt]
    
    # Manual Add-On: (Grounding Context): If there were no context items retrieved, add a warning. Otherwise, display it was retrieved. 
    add_check(
        "Grounding context",
        bool(context_items),
        "No retrieved context was available for this generation." if not context_items else "Retrieved context was available.",
    )

    # Manual Add-On: (No Stale Year Hashtags): Checks for the presence of year hashtags in the caption using a regular expression. If any year hashtags are found, the check fails and a warning is added indicating that the caption contains a year hashtag that should be removed.
    add_check(
        "No stale year hashtags",
        not re.search(r"#20\d{2}\b", caption),
        "Caption contains a year hashtag that should be removed.",
    )

    # Manual Add-On: (Dental Safety Wording): Checks for the presence of potentially unsafe or overconfident wording in the combined output (caption and video prompt) by looking for specific risky terms. 
    add_check(
        "Dental safety wording",
        not risky_hits,
        f"Potentially unsafe or overconfident wording found: {', '.join(risky_hits)}." if risky_hits else "No obvious diagnosis or guarantee language found.",
    )
    
    # Manual Add-On: (CTA Presence): Checks for the presence of a clear call to action (CTA) in the combined output by looking for specific CTA-related terms or the primary CTA text from the profile.
    add_check(
        "CTA present",
        any(term in combined_output for term in cta_terms) or (cta_text and cta_text in combined_output),
        "No clear booking, appointment, consultation, or question-based CTA found.",
    )

    # Manual Add-On: (Practice Relevance): Checks for the presence of practice-specific markers in the combined output to ensure that the generated content is relevant to the dental practice. It looks for mentions of the practice name, location, or services offered.
    add_check(
        "Practice relevance",
        bool(grounding_hits),
        "Output may be too generic; no practice, location, or service marker was detected.",
    )

    # Manual Add-On: (Retrieved Evidence Use): Uses the evidence_overlap score calculated by the context_term_overlap function to check for meaningful overlap between the generated output and the retrieved context. If there are no context items or if the overlap is less than 2 terms, the check fails and a warning is added indicating that the output appears disconnected from the retrieved context and should be reviewed for generic or unsupported wording.
    add_check(
        "Retrieved evidence use",
        not context_items or evidence_overlap >= 2,
        "Output appears disconnected from retrieved context; review for generic or unsupported wording.",
    )

    # Manual Add-On: (Video Prompt Completeness): Checks for the presence of key production details in the video prompt, such as vertical format, runtime, scene direction, audio/mood, and text overlays. If any of these elements are missing from the video prompt, the check fails.
    add_check(
        "Video prompt completeness",
        not missing_video_parts,
        f"Video prompt may be missing: {', '.join(missing_video_parts)}." if missing_video_parts else "Video prompt includes core production details.",
    )

    # Manual Add-On: (Fixed Schema Contract): Checks that the video prompt adheres to the fixed schema contract by verifying the presence of required section headings. If any required sections are missing from the video prompt, the check fails and a warning is added indicating which sections are missing.
    add_check(
        "Fixed schema contract",
        not missing_schema_sections,
        f"Video prompt is missing required section(s): {', '.join(missing_schema_sections)}."
        if missing_schema_sections
        else "Video prompt includes all required schema sections.",
    )

    # Manual Add-On: (Model-Safe Wording): Checks for the presence of potentially unsafe terms in the video prompt that could be inappropriate for patient-facing marketing content. If any unsafe terms are found, the check fails and a warning is added.
    add_check(
        "Model-safe constraints",
        not unsafe_model_hits,
        f"Video prompt includes potentially unsafe terms: {', '.join(unsafe_model_hits)}."
        if unsafe_model_hits
        else "Video prompt passed model-safe wording checks.",
    )

    # Finalize the overall status based on the checks performed. If there are no warnings, the status is "passed". If there are warnings but no failed checks, the status is "warning". If there are any failed checks, the status is "failed".
    status = "passed" if not warnings else "warning"
    if not caption.strip() or not video_prompt.strip():
        status = "failed"
        warnings.append("Caption or video prompt is empty.")

    # Returns the warnings 
    return {
        "status": status,
        "warnings": warnings,
        "checks": checks,
        "context_item_count": len(context_items),
        "evidence_overlap_terms": evidence_overlap if context_items else 0,
        "review_summary": "Ready with warnings to review." if warnings else "Ready to use.",
    }


@traceable(name= "[b7] Enforce Video Prompt Schema")
def enforce_video_prompt_schema(video_prompt):
    """
    Normalize output to the fixed eight-section video prompt contract.

    If the model drifts, this guard preserves heading order so downstream QA and
    external video-tool workflows remain stable.
    """

    # Initialize the list of canonical section names that define the fixed schema contract for video generation 
    canonical_sections = [
        "Objective",
        "Format/Duration",
        "Scene Breakdown",
        "Camera/Lighting/Motion",
        "Audio/Mood",
        "On-screen Text",
        "Final CTA Frame",
        "Negative Constraints",
    ]


    # Helper Function [1] (clear_section_text) 
    def clean_section_text(text):
        
        # Returns a section of the text without unnecessary formatting, whitespace, and other artifacts 
        return re.sub(r"^[\s*\-•_]+", "", text).strip()
    

    # Helper Function [2] (normalize_header_candidate)
    def normalize_header_candidate(text):
        # Normalize bold/markdown wrappers before section-header matching.
        normalized = text.strip()
        normalized = re.sub(r"^\s*[*_`#>\-]+\s*", "", normalized)
        normalized = re.sub(r"\s*[*_`]+\s*$", "", normalized)
        return normalized.strip()
    
    # Initializes lines by splitting the video prompt into individual lines and stripping any trailing whitespace from each line. This prepares the video prompt for processing and section extraction.
    lines = [line.rstrip() for line in video_prompt.splitlines()]

    # Initializes section blocks as a dictionary with canonical section names as keys and empty lists as values to store the lines of text corresponding to each section. It also initializes current_section to keep track of the currently active section while processing the lines, raw_text to hold the original video prompt for fallback purposes, and matched_any_section to track whether any valid section headers were recognized during processing.
    section_blocks = {name: [] for name in canonical_sections}

    # Initializes current_section and sets it as None for default 
    current_section = None

    # Initializes raw_text as the raw text that was extracted from the video prompt
    raw_text = video_prompt.strip()

    # Initializes matched_any_section and set default value as None 
    matched_any_section = False

    # Precompiles regular expression patterns for each canonical section to efficiently match section headers in the video prompt. The patterns are designed to recognize optional numbering (e.g., "1. ", "2) ") followed by the section name, allowing for flexible formatting while still accurately identifying the intended sections.
    section_patterns = {
        section: re.compile(
            rf"^\s*(?:\d+[.)]\s*)?{re.escape(section)}\s*:?\s*(.*)$",
            flags=re.IGNORECASE,
        )
        for section in canonical_sections
    }

    # Processes each line of the video prompt to identify section headers and organize the content into the corresponding sections. The function iterates through each line, stripping whitespace and checking for matches against the precompiled section header patterns. If a section header is recognized, it updates the current_section and stores any trailing content under that section. If a line does not match a section header but a current_section is active, it appends the line to the content of that section. Blank lines are preserved as separators within sections.
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_section:
                section_blocks[current_section].append("")
            continue

        matched_section = None
        trailing_content = ""
        candidate = normalize_header_candidate(stripped)

        # Check if the candidate line matches any of the canonical section headers using the precompiled regular expression patterns. If a match is found, it identifies the matched section and captures any trailing content on the same line after the section header. The matched section is then set as the current_section for subsequent lines until another section header is encountered.
        for section_name in canonical_sections:
            match = section_patterns[section_name].match(candidate)
            if match:
                matched_section = section_name
                trailing_content = (match.group(1) or "").strip()
                break

        # If a matched section header is found, update the current_section to the matched section and set matched_any_section to True. If there is any trailing content on the same line as the section header, clean it using the clean_section_text function and append it to the corresponding section block in the section_blocks dictionary. Then, continue to the next line without further processing.
        if matched_section:
            current_section = matched_section
            matched_any_section = True
            if trailing_content:
                section_blocks[current_section].append(clean_section_text(trailing_content))
            continue
        
        # If the line does not match a section header but there is an active current_section, clean the line using the clean_section_text function and append it to the corresponding section block in the section_blocks dictionary. This allows for the accumulation of content under the currently active section until another section header is encountered.
        if current_section:
            section_blocks[current_section].append(clean_section_text(stripped))

    # Fallback: if no headings were recognized but the model returned useful
    # free-form text, preserve it instead of replacing everything with blanks.
    if not matched_any_section and raw_text:
        section_blocks["Objective"].append(
            "Create a short, patient-safe dental marketing clip aligned to the provided topic and caption."
        )
        section_blocks["Format/Duration"].append("9:16 vertical, 8-12 seconds.")
        section_blocks["Scene Breakdown"].append(raw_text)
        section_blocks["Negative Constraints"].append(
            "No diagnosis language, no guaranteed outcomes, no unsupported promotions, no graphic dental imagery."
        )

    normalized_parts = []
    
    # Iterates through each canonical section name in the predefined order and constructs the normalized video prompt by combining the section headers with their corresponding content from the section_blocks dictionary. For each section, it checks if there are any lines of content; if so, it joins them together with newlines and adds the section header followed by the content to the normalized_parts list. If a section has no content, it adds a default message "Not specified." under that section header. Finally, it joins all the normalized parts together with double newlines to create the final normalized video prompt.
    for section_name in canonical_sections:
        body_lines = [line for line in section_blocks[section_name] if line.strip()]
        body = "\n".join(body_lines).strip() if body_lines else "Not specified."
        normalized_parts.append(f"{section_name}:\n{body}")

    return "\n\n".join(normalized_parts).strip()


### --- Main Autonomous Generation Workflow [c] --- ##

@traceable(name = "[c] Autonomous Generation Process")
def run_autonomous_gen(topic_override=None):
    """
    Run the full marketing generation workflow.

    Args:
        topic_override: Optional user-provided topic from the Streamlit app. If
            omitted, the model chooses a topic from the business profile.

    Returns:
        A dictionary containing the topic, final Instagram caption, engineered
        video prompt, retrieved context, source metadata, and profile snapshot.
        The UI stores this dictionary in `st.session_state.generation`.
    """
    profile = load_business_profile()
    business_profile = format_business_profile(profile)
    print("--- 🤖 Thinking of a topic... ---")

    # A user-supplied topic keeps the flow deterministic; otherwise m.int acts
    # autonomously and picks a dental marketing angle for the configured office.
    if topic_override:
        topic = topic_override.strip()
    else:
        topic_chain = get_topic_generation_chain()
        raw_topic = topic_chain.invoke({"business_profile": business_profile})
        topic = clean_model_output(raw_topic)

    print(f"--- 💡 Selected Topic: {topic} ---")

    print(f"--- 🔍 Retrieving and reranking context for '{topic}'... ---")
    context_items = retrieve_ranked_context(topic, profile)
    context = format_context_package(context_items)
    sources = [item["metadata"] for item in context_items]

    post_chain = get_post_generation_chain()
    raw_post = post_chain.invoke({
        "topic": topic,
        "context": context,
        "business_profile": business_profile,
    })
    final_post = clean_model_output(raw_post)

    video_context_items = retrieve_ranked_video_context(topic, profile)
    video_context = format_context_package(video_context_items, max_items=8)

    video_chain = get_video_generation_chain()
    raw_video_prompt = video_chain.invoke({
        "topic": topic,
        "caption": final_post,
        "context": video_context,
        "business_profile": business_profile,
    })
    video_prompt = clean_model_output(raw_video_prompt)
    video_prompt = enforce_video_prompt_schema(video_prompt)
    validation_report = build_validation_report(final_post, video_prompt, context_items, profile)

    return {
        "topic": topic,
        "caption": final_post,
        "video_prompt": video_prompt,
        "validation_report": validation_report,
        "context": context,
        "context_items": context_items,
        "video_context": video_context,
        "video_context_items": video_context_items,
        "sources": sources,
        "business_profile": profile,
    }


@traceable(name = " [TEST][c] Manual RAG Generation")
def manual_rag_generation(user_query): 
    """
    Developer/debug helper for checking retrieval quality.

    This is intentionally separate from the marketing generation workflow. It
    lets you ask questions like "What services does this office offer?" and
    inspect the raw context/sources before trusting those documents in captions.
    """
    profile = load_business_profile()
    business_profile = format_business_profile(profile)
    context_items = retrieve_ranked_context(user_query, profile)
    context_text = format_context_package(context_items)
    sources = [item["metadata"] for item in context_items]

    test_prompt = ChatPromptTemplate.from_template(
        "Business profile:\n{business_profile}\n\n"
        "Retrieved context blocks:\n{context}\n\n"
        "Answer the user's question using ONLY the retrieved context blocks above.\n"
        "Rules:\n"
        "- Cite the context block(s) you used with bracket citations like [1] or [2].\n"
        "- If the retrieved context is related but does not answer the question, say: "
        "'I found related context, but it does not directly answer the question.'\n"
        "- Do not use outside knowledge or guesses.\n"
        "- Keep the answer concise and factual.\n\n"
        "Question: {question}"
    )

    test_chain = test_prompt | llm | StrOutputParser()
    answer = test_chain.invoke({
        "context": context_text,
        "question": user_query,
        "business_profile": business_profile,
    })

    final_answer = clean_model_output(answer)
    grounding_report = build_answer_grounding_report(final_answer, context_items)

    return final_answer, context_text, sources, context_items, grounding_report


if __name__ == "__main__":
    result = run_autonomous_gen()
    print(f"\nTopic:\n{result['topic']}")
    print(f"\nCaption:\n{result['caption']}")
    print(f"\nVideo Prompt:\n{result['video_prompt']}")
