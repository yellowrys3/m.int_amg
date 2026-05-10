"""
Streamlit dashboard for the m.int dental marketing MVP.

The UI collects user choices, displays progress, and shows generated outputs.
Business logic stays in `main.py`, RAG rebuilding stays in `ingest.py`, and
open-web research stays in `snapshot.py`.
"""

import streamlit as st

from ingest import initialize_rag
from main import load_business_profile
from main import manual_rag_generation
from main import reset_vectorstore
from main import run_autonomous_gen
from snapshot import run_research_scout


# Page setup is kept at the top because Streamlit requires `set_page_config()`
# before other page elements are rendered.
st.set_page_config(
    page_title="m.int - Dental Marketing Generator",
    page_icon="logo2.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_global_styles():
    """Apply the visual system for the Streamlit shell."""
    st.markdown(
        """
        <style>
            :root {
                --mint: #0f9f7a;
                --mint-dark: #08745e;
                --ink: #18232d;
                --muted: #60717f;
                --line: #dbe5e1;
                --surface: #ffffff;
                --soft: #f6faf8;
                --accent: #275d8c;
            }

            .stApp {
                background:
                    linear-gradient(180deg, #f7fbf9 0%, #eef7f3 42%, #f8fafc 100%);
                color: var(--ink);
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"] {
                display: none;
            }

            [data-testid="stSidebar"] {
                background: #0d1f25;
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] * {
                color: #eaf5f0;
            }

            [data-testid="stSidebar"] .stRadio > label,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: rgba(234, 245, 240, 0.76);
            }

            [data-testid="stSidebar"] div[role="radiogroup"] label {
                padding: 0.72rem 0.8rem;
                border-radius: 0.55rem;
                margin-bottom: 0.25rem;
                border: 1px solid transparent;
            }

            [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
                background: rgba(255, 255, 255, 0.06);
                border-color: rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
                background: rgba(15, 159, 122, 0.18);
                border-color: rgba(77, 201, 164, 0.4);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1480px;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                border-color: var(--line);
                border-radius: 0.65rem;
                background: rgba(255, 255, 255, 0.9);
                box-shadow: 0 12px 32px rgba(24, 35, 45, 0.06);
            }

            .stButton > button,
            .stButton button,
            div[data-testid="stButton"] > button,
            div[data-testid="stButton"] button {
                border-radius: 0.55rem !important;
                border: 1px solid var(--mint) !important;
                background: var(--mint) !important;
                color: white !important;
                font-weight: 650;
                min-height: 2.65rem;
                box-shadow: none !important;
            }

            .stButton > button p,
            .stButton button p,
            div[data-testid="stButton"] > button p,
            div[data-testid="stButton"] button p,
            .stButton > button *,
            .stButton button *,
            div[data-testid="stButton"] > button *,
            div[data-testid="stButton"] button * {
                color: white !important;
            }

            .stButton > button:hover,
            .stButton button:hover,
            div[data-testid="stButton"] > button:hover,
            div[data-testid="stButton"] button:hover {
                border-color: var(--mint-dark) !important;
                background: var(--mint-dark) !important;
                color: white !important;
            }

            .stButton > button:disabled,
            .stButton button:disabled,
            div[data-testid="stButton"] > button:disabled,
            div[data-testid="stButton"] button:disabled {
                background: #d8e6e1 !important;
                border-color: #d8e6e1 !important;
            }

            .stButton > button:disabled *,
            .stButton button:disabled *,
            div[data-testid="stButton"] > button:disabled *,
            div[data-testid="stButton"] button:disabled * {
                color: #6b7b86 !important;
            }

            .stTextInput input,
            .stTextArea textarea {
                border-radius: 0.5rem;
                border-color: #cad8d3;
                background: #fbfdfc;
            }

            div[data-testid="stMetric"] {
                border: 1px solid var(--line);
                border-radius: 0.55rem;
                padding: 0.8rem 0.9rem;
                background: rgba(255, 255, 255, 0.7);
            }

            div[data-testid="stMetric"] [data-testid="stMetricLabel"],
            div[data-testid="stMetric"] [data-testid="stMetricValue"],
            div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
                color: var(--ink);
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) {
                border: 1px solid rgba(164, 214, 195, 0.48);
                background:
                    linear-gradient(145deg, rgba(31, 86, 73, 0.97), rgba(13, 52, 46, 0.98));
                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.16),
                    0 18px 42px rgba(13, 52, 46, 0.2);
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) h1,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) h2,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) h3,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) p,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) label,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) span {
                color: #eefbf5;
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) .stCaptionContainer,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) .quiet-note {
                color: rgba(238, 251, 245, 0.78);
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) .stTextInput input,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) .stTextArea textarea {
                background: #f8fffc;
                border-color: rgba(164, 214, 195, 0.72);
                color: var(--ink);
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) .stTabs button p,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker) .stTabs button[role="tab"] p {
                color: rgba(238, 251, 245, 0.78) !important;
            }

            div[data-testid="stTabs"] button,
            div[data-testid="stTabs"] button[role="tab"] {
                background: rgba(255, 255, 255, 0.64);
                border-radius: 0.45rem 0.45rem 0 0;
                font-weight: 650;
            }

            div[data-testid="stTabs"] button p,
            div[data-testid="stTabs"] button[role="tab"] p {
                color: #425563 !important;
            }

            div[data-testid="stTabs"] button[aria-selected="true"],
            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
                background: rgba(15, 159, 122, 0.1);
            }

            div[data-testid="stTabs"] button[aria-selected="true"] p,
            div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
                color: var(--mint-dark) !important;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
                background-color: var(--mint-dark);
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker)
                div[data-testid="stTabs"] button,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker)
                div[data-testid="stTabs"] button[role="tab"] {
                background: rgba(255, 255, 255, 0.1);
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker)
                div[data-testid="stTabs"] button[aria-selected="true"],
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker)
                div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
                background: rgba(255, 255, 255, 0.18);
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker)
                div[data-testid="stTabs"] button[aria-selected="true"] p,
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker)
                div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
                color: #eefbf5 !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"]:has(.metal-panel-marker)
                div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
                background-color: #6ab99a;
            }

            .hero-kicker {
                color: var(--mint-dark);
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .hero-title {
                color: var(--ink);
                font-size: 2.1rem;
                line-height: 1.15;
                font-weight: 800;
                margin: 0;
            }

            .hero-copy {
                color: var(--muted);
                font-size: 1rem;
                margin-top: 0.45rem;
                max-width: 760px;
            }

            .section-label {
                color: var(--muted);
                font-size: 0.84rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0;
            }

            .metal-panel-marker {
                color: #5d987f;
            }

            .quiet-note {
                color: var(--muted);
                font-size: 0.92rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(profile):
    """Render the persistent sidebar navigation and return the active view."""
    with st.sidebar:
        st.image("logo2.png", width=116)
        st.markdown("### m.int")
        st.caption("Dental marketing workspace")
        st.divider()

        nav = st.radio(
            "Navigation",
            ["⌂  Home", "⚙  Settings"],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Practice**")
        st.markdown(f"{profile['practice_name']}")
        st.caption(f"{profile['location']} | {profile['tone']}")

    return "settings" if "Settings" in nav else "home"


def render_header(profile):
    """Render the primary page heading and practice summary."""
    st.markdown('<div class="hero-kicker">Augmented Marketing Intelligence</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">Dental content generation, grounded in your practice data.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hero-copy">Create captions, video prompts, and QA reports for '
        f'{profile["practice_name"]}. Manage the knowledge base and research scout from one calmer workspace.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Practice", profile["practice_name"])
    metric_b.metric("Location", profile["location"])
    metric_c.metric("Services", len(profile.get("services", [])))


def render_data_management():
    """Render knowledge-base refresh controls."""
    with st.container(border=True):
        st.markdown(
            '<div class="section-label metal-panel-marker">Data Management</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Knowledge Base")
        st.caption("Rebuild the local vector database after changing PDFs, text files, or saved research.")

        if st.button("Refresh Knowledge Base", use_container_width=True):
            with st.status("Reading source files and updating ChromaDB...") as status:
                try:
                    # Release any Chroma connection opened by generation/RAG tests
                    # before deleting and rebuilding the persisted SQLite files.
                    reset_vectorstore()
                    initialize_rag()
                    reset_vectorstore()
                    status.update(label="Knowledge base synced.", state="complete")
                except Exception as error:
                    status.update(label="Knowledge base refresh failed.", state="error")
                    st.error(f"Refresh failed: {error}")


def render_research_scout():
    """Render the open-web research collection workflow."""
    with st.container(border=True):
        st.markdown('<div class="section-label">Research Scout</div>', unsafe_allow_html=True)
        st.subheader("Add New Research")
        st.caption("Save useful web research into the knowledge base, then refresh the database to embed it.")

        scout_topic = st.text_input(
            "Research topic",
            placeholder="e.g., Latest dental implant patient questions",
        )
        if st.button("Run Research Scout", use_container_width=True):
            if scout_topic:
                with st.spinner(f"Researching {scout_topic}..."):
                    try:
                        # Research Scout only writes new text files. The user still
                        # refreshes the knowledge base afterward to embed them.
                        saved_files = run_research_scout(scout_topic)
                        if saved_files:
                            st.success(f"Knowledge on '{scout_topic}' added.")
                            st.caption(f"Saved {len(saved_files)} research files.")
                            st.info("Refresh the knowledge base to sync new research into RAG.")
                        else:
                            st.warning(
                                "No usable research pages were saved. Try a more specific topic or broader wording."
                            )
                            skipped = getattr(run_research_scout, "last_skipped_reasons", {})
                            if skipped:
                                st.caption(f"Filtered results: {skipped}")
                    except RuntimeError as error:
                        st.error(str(error))
            else:
                st.warning("Please enter a topic for the Research Scout.")


def render_generation_workspace():
    """Render caption/video generation and developer RAG testing controls."""
    with st.container(border=True):
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        st.subheader("Generate Dental Marketing Content")
        reg_gen_tab, rag_test_tab = st.tabs(["Generation", "Developer RAG Test"])

        with reg_gen_tab:
            st.caption(
                "Generate a practice-specific Instagram caption and a production-ready prompt for external video tools."
            )
            topic_override = st.text_input(
                "Optional topic",
                placeholder="Leave blank for m.int to choose a dental marketing topic",
            )
            if st.button("Generate Caption + Video Prompt", use_container_width=True):
                with st.spinner("Building grounded marketing content..."):
                    # Store the complete result dictionary so tab switches and
                    # Streamlit reruns do not erase the latest generation.
                    st.session_state.generation = run_autonomous_gen(topic_override or None)
                    st.success("Generation complete.")

        with rag_test_tab:
            st.caption("Inspect retrieval quality before generating patient-facing content.")
            user_query = st.text_input(
                "RAG query",
                placeholder="e.g., What services does Butterfly Dental Care offer?",
            )
            if st.button("Test RAG Retrieval", use_container_width=True):
                if not user_query:
                    st.warning("Please enter a RAG query first.")
                else:
                    with st.spinner("Retrieving relevant information..."):
                        # This path is for debugging the local knowledge base; it
                        # does not create marketing copy.
                        answer, context, sources, context_items, grounding_report = manual_rag_generation(user_query)
                        st.session_state.rag_test = {
                            "answer": answer,
                            "context": context,
                            "sources": sources,
                            "context_items": context_items,
                            "grounding_report": grounding_report,
                        }
                    st.success("RAG retrieval complete.")

            if "rag_test" in st.session_state:
                rag_result = st.session_state.rag_test
                report = rag_result["grounding_report"]
                st.markdown("#### Retrieval Result")
                st.markdown(f"**Answer:** {rag_result['answer']}")
                status = report.get("status")
                if status == "grounded":
                    st.success(report.get("message"))
                elif status == "no_context":
                    st.info(report.get("message"))
                else:
                    st.warning(report.get("message"))

                with st.expander("Ranked Context Items"):
                    st.json(rag_result["context_items"])
                with st.expander("Sources"):
                    st.json(rag_result["sources"])
                with st.expander("Retrieved Context"):
                    st.text_area("Context", rag_result["context"], height=260)


def render_output_workspace():
    """Render generated copy, video prompt, validation details, and raw metadata."""
    with st.container(border=True):
        st.markdown(
            '<div class="section-label metal-panel-marker">Output</div>',
            unsafe_allow_html=True,
        )
        st.subheader("Output Workspace")

        if "generation" not in st.session_state:
            st.markdown(
                '<div class="quiet-note">No content generated yet. Choose a topic or let m.int pick one.</div>',
                unsafe_allow_html=True,
            )
            return

        result = st.session_state.generation
        post_tab, video_tab, validation_tab, raw_tab = st.tabs(
            ["Instagram Caption", "Video Prompt", "Quality Report", "Raw Data"]
        )

        with post_tab:
            st.markdown(f"**Topic Focus:** `{result['topic']}`")
            st.text_area("Caption Copy", result["caption"], height=380)

        with video_tab:
            st.text_area("Engineered Video Prompt", result["video_prompt"], height=420)

        with validation_tab:
            report = result.get("validation_report", {})
            status = report.get("status", "unknown")
            if status == "passed":
                st.success(report.get("review_summary", "Ready to use."))
            elif status == "failed":
                st.error(report.get("review_summary", "Generation needs review."))
            else:
                st.warning(report.get("review_summary", "Ready with warnings to review."))

            st.markdown("**Checks**")
            for check in report.get("checks", []):
                label = "PASS" if check.get("passed") else "REVIEW"
                st.write(f"{label}: {check.get('name')} - {check.get('note')}")

            if report.get("warnings"):
                with st.expander("Warnings"):
                    st.write("\n".join(report["warnings"]))

        with raw_tab:
            st.json(
                {
                    "topic": result["topic"],
                    "status": result.get("validation_report", {}).get("status", "RAG-Verified"),
                    "practice": result["business_profile"]["practice_name"],
                    "sources": result["sources"],
                    "context_items": result.get("context_items", []),
                }
            )


def render_home(profile):
    """Render the main production workspace."""
    render_header(profile)
    st.write("")

    control_col, output_col = st.columns([0.92, 1.45], gap="large")
    with control_col:
        render_data_management()
        st.write("")
        render_research_scout()
        st.write("")
        render_generation_workspace()

    with output_col:
        render_output_workspace()


def render_settings(profile):
    """Render lightweight settings and operational context."""
    render_header(profile)
    st.write("")

    profile_col, runtime_col = st.columns([1.05, 0.95], gap="large")
    with profile_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
            st.subheader("Practice Profile")
            st.write(f"**Audience:** {profile.get('audience')}")
            st.write(f"**Primary CTA:** {profile.get('primary_cta')}")
            st.write(f"**Website:** {profile.get('website')}")
            with st.expander("Services"):
                st.write(", ".join(profile.get("services", [])))
            with st.expander("Content Guardrails"):
                for guardrail in profile.get("content_guardrails", []):
                    st.write(f"- {guardrail}")

    with runtime_col:
        with st.container(border=True):
            st.markdown('<div class="section-label">Runtime</div>', unsafe_allow_html=True)
            st.subheader("Local Model")
            st.info("Ensure Ollama is running DeepSeek locally before generating.")
            st.caption(
                "The settings page is intentionally lightweight for now; profile editing can be added here later."
            )


apply_global_styles()
business_profile = load_business_profile()
active_view = render_sidebar(business_profile)

if active_view == "settings":
    render_settings(business_profile)
else:
    render_home(business_profile)
