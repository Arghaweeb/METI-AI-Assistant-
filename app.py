import streamlit as st
import os
import sys
from datetime import datetime
import time
from dotenv import load_dotenv

# Import your RAG system components
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="METI Committee Information Agent",
    page_icon="assets/agile_energy_logo_v2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #000000 0%, #1AE315 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .committee-card {
        background: black;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1AE315;
        margin-bottom: 1rem;
    }
    
    .query-input {
        background: black;
        border: 2px solid #e9ecf;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .source-doc {
        background: #000000;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
    }
    
    .stAlert > div {
        border-radius: 8px;
    }
    
    .metric-card {
        background: black;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    /* --- BUTTON --- */
    button[kind="primary"] {
        background: linear-gradient(135deg, #000000 0%, #1AE315 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: bold;
        transition: 0.3s ease;
    }

    button[kind="primary"]:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }

    /* --- SLIDER TRACK --- */
    div[data-testid="stSlider"] > div > div > div {
        background: linear-gradient(135deg, #000000 0%, #1AE315 100%) !important;
        height: 6px !important;
        border-radius: 5px !important;
    }

    /* --- SLIDER THUMB --- */
    div[data-testid="stSlider"] > div > div > div > div {
        background-color: white !important;
        border: 2px solid #1AE315 !important;
        height: 20px !important;
        width: 20px !important;
        margin-top: -7px !important;
        border-radius: 50% !important;
    }

    /* --- SLIDER LABEL (Min/Max Text) --- */
    div[data-testid="stSlider"] label {
        color: white !important;
        font-weight: bold;
    }

    /* --- SLIDER VALUE TOOLTIP (Number on Top) --- */
    div[data-testid="stSlider"] span {
        color: #1AE315 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Define prompt templates (same as in your RAG system)
COMPREHENSIVE_PROMPT = """You are an expert assistant specializing in Japan's electricity and energy policy, with access to detailed information from 8 key METI (Ministry of Economy, Trade and Industry) committee meetings held in 2025.

## Your Knowledge Base
Your knowledge base contains official meeting documents from these committees:

1. **Subcommittee on Basic Electricity and Gas Policy** (電力・ガス基本政策小委員会)
   - Meetings: 85th-87th committee meetings
   - Focus: Fundamental electricity and gas policy frameworks, regulatory reforms, market mechanisms

2. **Subcommittee on Large-Scale Introduction of Renewable Energy and Next-Generation Electricity Networks** (再生可能エネルギー大量導入・次世代電力ネットワーク小委員会)
   - Meetings: 72nd-74th committee meetings
   - Focus: Large-scale renewable energy deployment, grid modernization, network infrastructure

3. **Next Generation Power System Working Group** (次世代電力系統ワーキンググループ)
   - Meetings: 1st-2nd sessions in 2025
   - Focus: Advanced power system technologies, grid flexibility, smart grid implementation

4. **Study Group on Next-Generation Distributed Power Systems** (次世代の分散型電力システムに関する検討会)
   - Meetings: 12th study group meeting
   - Focus: Distributed energy resources, microgrids, decentralized power systems

5. **Watt Bit Collaboration Public-Private Forum** (ワット・ビット連携官民懇談会)
   - Meetings: 1st-3rd sessions in 2025
   - Focus: Digital transformation in energy sector, data utilization, public-private partnerships

6. **Carbon Management Subcommittee** (カーボンマネジメント小委員会)
   - Meetings: 9th meeting
   - Focus: Carbon management strategies, decarbonization policies, emission reduction measures

7. **Study Group on the Status of Simultaneous Markets** (同時市場の在り方等に関する検討会)
   - Meetings: 13th-17th meetings
   - Focus: Electricity market design, market coupling, simultaneous market operations

8. **Committee on Adjustment Capacity and Supply-Demand Balance Evaluation** (調整力及び需給バランス評価等に関する委員会)
   - Focus: Grid balancing services, supply-demand management, adjustment capacity mechanisms

## Response Requirements
- ✅ Cite specific committee meetings and sources
- ✅ Provide context about which committee discussed the topic
- ✅ Use official METI terminology and policy language
- ✅ Include Japanese terms when appropriate for authenticity
- ✅ Be accurate and precise, avoiding speculation beyond documented information
- ✅ If information is not available in the documents, clearly state this limitation
- ✅ Maintain professional and authoritative tone appropriate for government policy discussions
## Context Information
Based on the following retrieved documents from METI committee meetings:

{context}

## User Question
{question}

## Response
Please provide a comprehensive answer following the guidelines above

**IMPORTANT: If the {question} is written in English, answer the {question} in English. If the {question} is written in Japanese, answer it in Japanese as from the documents."""

SIMPLE_PROMPT = """
You are a METI energy policy expert with access to official 2025 committee meeting documents.
You will answer questions based only on the provided context.

=== CONTEXT ===
{context}

=== USER QUESTION ===
{question}

=== INSTRUCTIONS ===
- Answer using only the above context
- Cite specific committee names and meeting numbers where possible
- Include Japanese terminology for technical terms when appropriate
- Maintain a formal, policy-expert tone
- ⚠️ VERY IMPORTANT: If the user question is written in **English**, respond in **English**.  
  If the user question is written in **Japanese**, respond in **Japanese** based on the documents.
- If no information is found in the context, explicitly say: "The provided documents do not contain information related to this question."

=== ANSWER ===
"""

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching"""
    try:
        # AWS + Pinecone Configs
        AWS_REGION = st.secrets["AWS_REGION"]
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
        PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
        PINECONE_NAMESPACE = st.secrets["PINECONE_NAMESPACE"]
        
        # Validate environment variables
        if not all([AWS_REGION, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
            st.error("Missing required environment variables. Please check your .env file.")
            return None
        
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize embeddings
        embedding = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
            region_name=AWS_REGION
        )
        
        # Initialize vector store
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embedding,
            text_key="text",
            namespace=os.getenv("PINECONE_NAMESPACE")
        )
        
        # Initialize LLM
        llm = ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region_name=AWS_REGION
        )
        
        return {
            'vectorstore': vectorstore,
            'llm': llm,
            'pc': pc,
            'index': index
        }
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def create_qa_chain(rag_system, prompt_type="comprehensive", retrieval_k=5):
    """Create QA chain with specified prompt type"""
    try:
        if prompt_type == "comprehensive":
            template = COMPREHENSIVE_PROMPT
        else:
            template = SIMPLE_PROMPT
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=rag_system['llm'],
            chain_type="stuff",
            retriever=rag_system['vectorstore'].as_retriever(search_kwargs={"k": retrieval_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

def query_system(question, prompt_type="comprehensive", retrieval_k=5):
    """Query the RAG system"""
    try:
        if not st.session_state.rag_system:
            st.error("RAG system not initialized. Please check your configuration.")
            return None
        
        qa_chain = create_qa_chain(st.session_state.rag_system, prompt_type, retrieval_k)
        if not qa_chain:
            return None
        
        with st.spinner("Searching through METI committee documents..."):
            result = qa_chain.invoke({"query": question})
        
        return result
    
    except Exception as e:
        st.error(f"Error querying system: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ METI Committee Information Assistant</h1>
        <p>Official information from 8 key METI committee meetings held in 2025 (Comming Soon: 2023,2024)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.system_initialized:
        with st.spinner("Initializing METI Committee Search System..."):
            st.session_state.rag_system = initialize_rag_system()
            if st.session_state.rag_system:
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system. Please check your configuration.")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Prompt type selection
        prompt_type = st.selectbox(
            "Select Response Style:",
            ["comprehensive", "simple"],
            help="Comprehensive provides detailed context, Simple gives concise answers"
        )
        
        # Number of documents to retrieve
        retrieval_k = st.slider(
            "Documents to Retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant documents to use for answering. Higher the document you choose to retrieve the longer it will take to generate the answer"
        )
        
        st.markdown("---")
        
        # Committee information
        st.header("📋 Committee Coverage (Information on More Committee Meetings will be released soon )")
        
        committees = [
            ("Basic Electricity & Gas Policy", "電力・ガス基本政策小委員会", "85th-87th meetings"),
            ("Renewable Energy & Networks", "再生可能エネルギー大量導入・次世代電力ネットワーク小委員会", "72nd-74th meetings"),
            ("Next Generation Power System", "次世代電力系統ワーキンググループ", "1st-2nd sessions"),
            ("Distributed Power Systems", "次世代の分散型電力システムに関する検討会", "12th meeting"),
            ("Watt Bit Collaboration", "ワット・ビット連携官民懇談会", "1st-3rd sessions"),
            ("Carbon Management", "カーボンマネジメント小委員会", "9th meeting"),
            ("Simultaneous Markets", "同時市場の在り方等に関する検討会", "13th-17th meetings"),
            ("Adjustment Capacity", "調整力及び需給バランス評価等に関する委員会", "2025 meetings")
        ]
        
        for name_en, name_jp, meetings in committees:
            st.markdown(f"""
            <div class="committee-card">
                <strong>{name_en}</strong><br>
                <small>{name_jp}</small><br>
                <em>{meetings}</em>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Language info
        st.header("🌐 Language Support")
        st.info("Ask questions in English or Japanese (日本語) - the system will respond in the same language!")
        
        # Example questions
        st.header("💡 Example Questions")
        example_questions = [
            "What external changes are impacting Japan's electricity system?",
            "日本の再生可能エネルギーの現状は？",
            "What are the main challenges in grid modernization?",
            "電力市場の改革について教えてください",
            "How is Japan addressing carbon management?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"Try: {question[:30]}...", key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Question")
        
        # Question input
        if 'current_question' in st.session_state:
            question = st.text_area(
                "Enter your question about METI committees:",
                value=st.session_state.current_question,
                height=100,
                placeholder="例: 日本の電力システムに影響を与える外部変化は何ですか？\nExample: What external changes are impacting Japan's electricity system?"
            )
            del st.session_state.current_question
        else:
            question = st.text_area(
                "Enter your question about METI committees:",
                height=100,
                placeholder="例: 日本の電力システムに影響を与える外部変化は何ですか？\nExample: What external changes are impacting Japan's electricity system?"
            )
        
        # Query button
        if st.button("🔍 Search METI Documents", type="primary"):
            if question.strip():
                # Record query time
                query_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Query the system
                result = query_system(question, prompt_type, retrieval_k)
                
                if result:
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': result['result'],
                        'source_documents': result['source_documents'],
                        'timestamp': query_time,
                        'prompt_type': prompt_type,
                        'retrieval_k': retrieval_k
                    })
                    
                    st.success("✅ Query completed successfully!")
                else:
                    st.error("❌ Query failed. Please try again.")
            else:
                st.warning("⚠️ Please enter a question.")
    
    with col2:
        st.header("📊 System Status")
        
        # System metrics
        if st.session_state.rag_system:
            st.markdown("""
            <div class="metric-card">
                <h3>🟢 System Online</h3>
                <p>RAG system operational</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Query statistics
        total_queries = len(st.session_state.chat_history)
        st.metric("Total Queries", total_queries)
        
        if st.session_state.chat_history:
            latest_query = st.session_state.chat_history[-1]['timestamp']
            st.metric("Last Query", latest_query)
    
    # Display results
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("📋 Query Results")
        
        # Display latest result
        latest = st.session_state.chat_history[-1]
        
        # Question
        st.subheader("❓ Question:")
        st.write(latest['question'])
        
        # Answer
        st.subheader("💡 Answer:")
        st.write(latest['answer'])
        
        # Source documents
        if latest['source_documents']:
            st.subheader("Source Documents:")
            
            for i, doc in enumerate(latest['source_documents']):
                with st.expander(f"Document {i+1}"):
                    st.markdown(f"""
                    <div class="source-doc">
                        <strong>Content Preview:</strong><br>
                        {doc.page_content[:500]}...
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if doc.metadata:
                        st.json(doc.metadata)
        
        # Query details
        with st.expander("Query Details"):
            st.write(f"**Prompt Type:** {latest['prompt_type']}")
            st.write(f"**Documents Retrieved:** {latest['retrieval_k']}")
            st.write(f"**Timestamp:** {latest['timestamp']}")
    
    # Chat history
    if len(st.session_state.chat_history) > 1:
        st.markdown("---")
        st.header("📝 Query History")
        
        for i, entry in enumerate(reversed(st.session_state.chat_history[:-1])):
            with st.expander(f"Query {len(st.session_state.chat_history) - i - 1}: {entry['question'][:50]}..."):
                st.write(f"**Question:** {entry['question']}")
                st.write(f"**Answer:** {entry['answer']}")
                st.write(f"**Timestamp:** {entry['timestamp']}")
    
    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()