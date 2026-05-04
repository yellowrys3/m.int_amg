import streamlit as st
from main import run_autonomous_gen
from main import manual_rag_generation 
from snapshot import run_research_scout
from ingest import intialize_rag
import os


# 1. Professional Page Config
st.set_page_config(
    page_title="m.int - Autonomous Marketing Generation",
    page_icon="logo2.png", # Minty fresh icon
    layout="wide"
)

# 2. Custom "Minty" Header
st.markdown("""
    <style>
    .main { background-color: #f0fdf4; } /* Very light mint background */
    .stButton>button { background-color: #10b981; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# Top Navigation Bar
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo2.png", width=1500)

with col2:
    st.markdown("### **Augmented Marketing Intelligence**")
    st.caption("Precision AI for Modern Dental Practices")

st.divider()


# 3. Sidebar with a "Control Panel" vibe
with st.sidebar:
    st.header("Control Center")
    st.info("Ensure Ollama is running DeepSeek locally before generating.")
    
    st.subheader("📁 Data Management")
    if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
        with st.status("Reading PDFs & Updating ChromaDB...") as status:
            intialize_rag()
            status.update(label="Knowledge Base Synced!", state="complete")
    
    st.divider() 
    st.subheader("🧠 Research Scout")        
    
    scout_topic = st.text_input("Enter a topic for the Research Scout:", placeholder="e.g., Latest dental technology trends")
    if st.button("🔍 Run Research Scout", use_container_width=True):
        if scout_topic:
            with st.spinner(f"🔍 Researching {scout_topic}..."):
                run_research_scout(scout_topic)
                st.success(f"Knowledge on '{scout_topic}' added to the knowledge base!")
                st.info("⚠️ Remember to click 'Refresh Knowledge Base' to sync with m.int's RAG system. ")
        else:
            st.warning("Please enter a topic for the Research Scout.")

# 4. Main Generation Dashboard
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    with st.container(border=True):

        st.subheader("Generate Autonomous Content")
        reg_gen_tab, rag_test_tab = st.tabs(["🚀 Run Generation", "🔍 Test RAG"])
    
        with reg_gen_tab:
            with st.container(border=True):
                  # Ensure the container has a minimum height for better UX
                st.write("Click below to start the regular autonomous pipeline.")
                if st.button("🚀 Launch M.int Generation", use_container_width=True):
                    with st.spinner("🌿 M.int is browsing your data..."):
                        st.session_state.topic, st.session_state.post, st.session_state.video = run_autonomous_gen()
                        st.success("Generation Complete!")
            

        with rag_test_tab: 
            st.write("Test the RAG retrieval with a custom query.")
            user_query = st.text_input("Enter a query to test RAG retrieval:", placeholder="e.g., What are the latest dental trends?")
            if st.button("🔍 Test RAG Retrieval", use_container_width=True):
                with st.spinner("Retrieving relevant information..."):
                    answer, context = manual_rag_generation(user_query)
                    st.subheader("RAG Retrieval Result")
                    st.markdown(f"**Answer:** {answer}")
                    with st.expander("View Retrieved Context"):
                        st.text_area("Context", context, height=300)

        st.markdown('<div style = "min-height: 120px;">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)  # Close the container div    


with right_col:
       
    st.subheader("Output Workspace")
    if 'topic' in st.session_state:
        # Using Tabs for a clean workspace
        post_tab, video_tab, raw_tab = st.tabs(["📸 Instagram", "🎬 Video Script", "📋 Raw Data"])
        
        with post_tab:
            st.markdown(f"**Topic Focus:** `{st.session_state.topic}`")
            st.text_area("Caption Copy", st.session_state.post, height=350)
            st.button("📋 Copy Caption")
            
        with video_tab:
            st.info(st.session_state.video)
            
        with raw_tab:
            st.json({"topic": st.session_state.topic, "status": "RAG-Verified"})
    else:
        st.write("No content generated yet. Hit the launch button to begin!")

