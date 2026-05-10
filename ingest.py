"""
Knowledge-base ingestion for the m.int dental marketing MVP.

This module turns raw office and research materials into a local Chroma vector
database. The generator in `main.py` relies on this database to ground captions
and video prompts in real practice information rather than generic model memory.

Expected source files live in `knowledge_base/`:
- Practice/profile content, such as scraped office website pages.
- Trend research files written by `snapshot.py` through crawl4ai.
- Optional PDFs, such as service documents or marketing briefs.
"""


import os 
import re
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings   
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv 


load_dotenv()


# [DEBUG/TEST] Environment Verification Test
if os.getenv("LANGSMITH_API_KEY"):
    print("✅ LangSmith API Key Loaded")
else:
    print("❌ LangSmith API Key Missing")


HEADER_FIELDS = {
    "TYPE": "knowledge_type",
    "TOPIC": "topic",
    "SOURCE": "source_url",
    "CRAWLED_AT": "crawled_at",
}
CHROMA_DB_DIR = Path("./chroma_db")


def make_writable(path):
    """
    Best-effort permission repair for local Chroma files before deletion.

    Failed or interrupted SQLite writes can occasionally leave files with
    restrictive permissions. Making the tree writable before `rmtree()` keeps
    refreshes from getting stuck on stale local artifacts.
    """
    try:
        path.chmod(0o700 if path.is_dir() else 0o600)
    except OSError:
        pass


def remove_existing_chroma_db():
    """
    Delete the persisted Chroma directory before a full rebuild.

    `app.py` releases the active vectorstore first. This function then removes
    the old on-disk index so the rebuilt database exactly matches the current
    knowledge_base contents.
    """
    if not CHROMA_DB_DIR.exists():
        return

    for db_path in CHROMA_DB_DIR.rglob("*"):
        make_writable(db_path)
    make_writable(CHROMA_DB_DIR)
    shutil.rmtree(CHROMA_DB_DIR)


def parse_research_headers(text):
    """
    Extract metadata headers written by Research Scout.

    crawl4ai snapshots start with simple `KEY: value` lines. Pulling those into
    Chroma metadata lets retrieval filter by topic, source URL, and crawl time
    instead of treating that operational metadata as ordinary body text.
    """
    metadata = {}
    body_lines = []
    in_header = True

    for line in text.splitlines():
        if in_header and not line.strip():
            in_header = False
            continue

        if in_header and ":" in line:
            key, value = line.split(":", 1)
            field_name = HEADER_FIELDS.get(key.strip().upper())
            if field_name:
                metadata[field_name] = value.strip()
                continue

        in_header = False
        body_lines.append(line)

    return metadata, "\n".join(body_lines)


def classify_knowledge_source(source_path, header_metadata):
    """
    Classify a document as practice context or trend research.

    Research Scout files are timestamped `research_...txt` files and may also
    declare `TYPE: trend_research`. Everything else is treated as office/profile
    material so practice facts can be retrieved separately from broad research.
    """
    declared_type = header_metadata.get("knowledge_type")
    if declared_type in {"practice_profile", "trend_research", "prompt_playbook"}:
        return declared_type

    source_name = Path(source_path).name
    if "research_" in source_name:
        return "trend_research"
    if "prompt_playbook_" in source_name:
        return "prompt_playbook"
    return "practice_profile"


def extract_source_title(text, source_path):
    """
    Derive a readable title for source displays in the RAG/debug UI.

    Prefer the first Markdown heading from crawled content. If none exists, use
    the filename stem so every context item still has a compact label.
    """
    heading_match = re.search(r"^\s{0,3}#{1,3}\s+(.+)$", text, flags=re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()[:120]

    return Path(source_path).stem.replace("_", " ").title()


def clean_document_text(text):
    """
    Remove low-value crawl noise before chunking and embedding.

    The goal is not perfect web extraction; it is to keep obvious navigation,
    placeholder images, social links, and repeated whitespace from dominating
    similarity search results.
    """
    cleaned = re.sub(r"!\[[^\]]*\]\(data:image/svg\+xml[^)]*\)", " ", text)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]{0,40}\]\([^)]*(facebook|instagram|x\.com|tel:|mailto:)[^)]*\)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(read more|read less|skip to content|load more|feedback)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def prepare_documents(documents):
    """
    Normalize loaded LangChain documents before splitting.

    This enriches metadata, strips Research Scout headers out of page content,
    cleans noisy crawl text, and drops tiny documents that would create weak
    vector chunks.
    """
    prepared_docs = []

    for document in documents:
        source_path = document.metadata.get("source", "")
        header_metadata, body_text = parse_research_headers(document.page_content)
        knowledge_type = classify_knowledge_source(source_path, header_metadata)
        cleaned_text = clean_document_text(body_text)

        if len(cleaned_text.split()) < 25:
            continue

        document.page_content = cleaned_text
        document.metadata.update(header_metadata)
        document.metadata["knowledge_type"] = knowledge_type
        document.metadata["source_path"] = source_path
        document.metadata["source_url"] = document.metadata.get("source_url", "")
        document.metadata["topic"] = document.metadata.get("topic", "")
        document.metadata["crawled_at"] = document.metadata.get("crawled_at", "")
        document.metadata["source_title"] = extract_source_title(cleaned_text, source_path)
        prepared_docs.append(document)

    return prepared_docs


def split_documents_by_type(documents):
    """
    Split practice and research documents with slightly different chunk sizes.

    Practice pages benefit from tighter chunks because office facts are usually
    short and specific. Research pages can be larger so the generator receives
    enough surrounding context for trend summaries.
    """
    practice_docs = [doc for doc in documents if doc.metadata.get("knowledge_type") == "practice_profile"]
    research_docs = [doc for doc in documents if doc.metadata.get("knowledge_type") == "trend_research"]
    playbook_docs = [doc for doc in documents if doc.metadata.get("knowledge_type") == "prompt_playbook"]

    practice_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        add_start_index=True,
    )
    research_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100,
        chunk_overlap=180,
        add_start_index=True,
    )
    playbook_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=140,
        add_start_index=True,
    )

    split_docs = (
        practice_splitter.split_documents(practice_docs)
        + research_splitter.split_documents(research_docs)
        + playbook_splitter.split_documents(playbook_docs)
    )
    chunk_counts = {}
    filtered_docs = []

    for split_doc in split_docs:
        if len(split_doc.page_content.split()) < 35:
            continue

        source_path = split_doc.metadata.get("source_path") or split_doc.metadata.get("source", "")
        chunk_index = chunk_counts.get(source_path, 0)
        chunk_counts[source_path] = chunk_index + 1
        split_doc.metadata["chunk_index"] = chunk_index
        filtered_docs.append(split_doc)

    return filtered_docs


def initialize_rag():
    """
    Rebuild the local Retrieval-Augmented Generation index.

    The refresh process is intentionally full-rebuild rather than append-only.
    That prevents duplicate chunks from piling up when the Streamlit user clicks
    "Refresh Knowledge Base" after adding crawl4ai research snapshots.

    Returns:
        None. The side effect is a refreshed `./chroma_db` directory containing
        embedded chunks and metadata used by `main.py`.
    """
    
    # Load text and PDF sources separately so future file-type handling can be
    # customized without changing the rest of the ingestion pipeline.
    txt_loader = DirectoryLoader(  # TXT file loader 
        "./knowledge_base", 
        glob="**/*.txt",
        loader_cls=TextLoader  
    )

    pdf_loader = DirectoryLoader(  # PDF file loader 
        "./knowledge_base", 
        glob="**/*.pdf",
        loader_cls=PyPDFLoader  
    )

    raw_documents = txt_loader.load() + pdf_loader.load()
    documents = prepare_documents(raw_documents)
    
    # Basic guardrail for empty or misconfigured knowledge_base folders.
    if documents:
        print(f"[DEBUG | SUCCESS] Loaded {len(documents)} cleaned documents from the knowledge base. ")
    else: 
        print("[DEBUG | ERROR] No documents found in the knowledge base. Please check the directory and file formats. ")
        return None


    # Split long pages into retrievable chunks. The overlap preserves context
    # across section boundaries, which helps when service descriptions span
    # multiple paragraphs.
    split_docs = split_documents_by_type(documents)

    # Confirm splitting succeeded before doing the more expensive embedding step.
    if split_docs:
        print(f"[DEBUG | SUCCESS] Split documents into {len(split_docs)} chunks for better processing. ")
    else: 
        print("[DEBUG | ERROR] Document splitting failed. Please check the text splitter configuration. ")
        return None
    

    # Recreate Chroma from scratch so refreshes reflect the current
    # knowledge_base exactly and do not keep stale/deleted documents.
    embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")  # Use a smaller model for faster embedding generation
    remove_existing_chroma_db()

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR) # Saves to a local directory
    )   
    print("[DEBUG | SUCCESS] Vector database created and persisted locally at './chroma_db'. ")


def intialize_rag():
    """
    Backward-compatible alias for the original misspelled function name.

    Existing imports or notebooks may still call `intialize_rag()`. Keeping this
    wrapper avoids breaking those callers while new code uses `initialize_rag()`.
    """
    return initialize_rag()


if __name__ == "__main__":
    initialize_rag() 
