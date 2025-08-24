import streamlit as st
import os
import sys
from datetime import datetime
import time
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

# Import your RAG system components
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()


def create_presigned_pdf_link(s3_uri, page_number=None, expiration=3600):
    """Create a presigned URL for the PDF document"""
    try:
        # Initialize S3 client using your AWS credentials
        s3_client = boto3.client(
            's3', 
            region_name=st.secrets["AWS_REGION"],
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"]
        )
        
        # Extract bucket and key from S3 URI
        s3_parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        object_key = s3_parts[1]
        
        # Generate presigned URL (valid for 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
             Params={
                 'Bucket': bucket_name, 
                 'Key': object_key,
                 'ResponseContentDisposition': 'inline',
                 'ResponseContentType': 'application/pdf'
            },
            ExpiresIn=expiration
        )
        
        # Extract filename for display
        filename = object_key.split("/")[-1]
        
        # Create link with page reference if available
        if page_number:
            return f"[📄 {filename} (Page {page_number})]({presigned_url})"
        else:
            return f"[📄 {filename}]({presigned_url})"
            
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return f"📄 {s3_uri} (Link unavailable)"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"📄 {s3_uri}"

# Language configurations
LANGUAGES = {
    "en": {
        "title": "METI Committee Information Agent",
        "subtitle": "Official information from 8 key METI committee meetings held in 2025",
        "config_header": "🎛️ Configuration",
        "response_style": "Select Response Style:",
        "documents_retrieve": "Documents to Retrieve:",
        "committee_coverage": "📋 Committee Coverage",
        "language_support": "🌐 Language Support",
        "language_support_text": "Ask questions in English or Japanese (日本語) - the system will respond in the same language!",
        "example_questions": "💡 Example Questions",
        "ask_question": "💬 Ask Your Question",
        "question_placeholder": "例: 日本の電力システムに影響を与える外部変化は何ですか？\nExample: What external changes are impacting Japan's electricity system?",
        "search_button": "🔍 Search METI Documents",
        "system_status": "📊 System Status",
        "system_online": "🟢 System Online",
        "system_operational": "RAG system operational",
        "total_queries": "Total Queries",
        "last_query": "Last Query",
        "query_results": "📋 Query Results",
        "question_label": "❓ Question:",
        "answer_label": "💡 Answer:",
        "source_documents": "📚 Source Documents:",
        "query_details": "Query Details",
        "query_history": "📝 Query History",
        "clear_history": "🗑️ Clear History",
        "content_preview": "Content Preview:",
        "prompt_type": "Prompt Type:",
        "documents_retrieved": "Documents Retrieved:",
        "timestamp": "Timestamp:",
        "comprehensive": "Comprehensive",
        "simple": "Simple",
        "comprehensive_help": "Comprehensive provides detailed context, Simple gives concise answers",
        "retrieval_help": "Number of relevant documents to use for answering. Higher the document you choose to retrieve the longer it will take to generate the answer",
        "query_success": "✅ Query completed successfully!",
        "query_failed": "❌ Query failed. Please try again.",
        "enter_question": "⚠️ Please enter a question.",
        "initializing": "Initializing METI Committee Search System...",
        "system_initialized": "✅ System initialized successfully!",
        "system_failed": "❌ Failed to initialize system. Please check your configuration.",
        "searching": "Searching through METI committee documents...",
        "coming_soon": "(Coming Soon: 2023, 2024)",
        "theme_mode": "🎨 Theme Mode",
        "dark_mode": "🌙 Dark Mode",
        "light_mode": "☀️ Light Mode"
    },
    "ja": {
        "title": "METI委員会情報エージェント",
        "subtitle": "2025年に開催された8つの主要METI委員会からの公式情報",
        "config_header": "🎛️ 設定",
        "response_style": "応答スタイルを選択:",
        "documents_retrieve": "取得する文書数:",
        "committee_coverage": "📋 委員会カバレッジ",
        "language_support": "🌐 言語サポート",
        "language_support_text": "英語または日本語で質問してください - システムは同じ言語で回答します！",
        "example_questions": "💡 質問例",
        "ask_question": "💬 質問をする",
        "question_placeholder": "例: 日本の電力システムに影響を与える外部変化は何ですか？\n例: 再生可能エネルギーの導入課題について教えてください",
        "search_button": "🔍 METI文書を検索",
        "system_status": "📊 システム状況",
        "system_online": "🟢 システム稼働中",
        "system_operational": "RAGシステム動作中",
        "total_queries": "総クエリ数",
        "last_query": "最終クエリ",
        "query_results": "📋 クエリ結果",
        "question_label": "❓ 質問:",
        "answer_label": "💡 回答:",
        "source_documents": "📚 参考文書:",
        "query_details": "クエリ詳細",
        "query_history": "📝 クエリ履歴",
        "clear_history": "🗑️ 履歴をクリア",
        "content_preview": "内容プレビュー:",
        "prompt_type": "プロンプトタイプ:",
        "documents_retrieved": "取得文書数:",
        "timestamp": "タイムスタンプ:",
        "comprehensive": "包括的",
        "simple": "シンプル",
        "comprehensive_help": "包括的は詳細なコンテキストを提供し、シンプルは簡潔な回答を提供します",
        "retrieval_help": "回答に使用する関連文書の数。取得する文書を多く選ぶほど、回答の生成に時間がかかります",
        "query_success": "✅ クエリが正常に完了しました！",
        "query_failed": "❌ クエリが失敗しました。もう一度お試しください。",
        "enter_question": "⚠️ 質問を入力してください。",
        "initializing": "METI委員会検索システムを初期化中...",
        "system_initialized": "✅ システムが正常に初期化されました！",
        "system_failed": "❌ システムの初期化に失敗しました。設定を確認してください。",
        "searching": "METI委員会文書を検索中...",
        "coming_soon": "（近日公開: 2023年、2024年）",
        "theme_mode": "🎨 テーマモード",
        "dark_mode": "🌙 ダークモード",
        "light_mode": "☀️ ライトモード"
    }
}

# Committee information in both languages
COMMITTEES_INFO = {
    "en": [
        ("Basic Electricity & Gas Policy", "電力・ガス基本政策小委員会"),
        ("Renewable Energy & Networks", "再生可能エネルギー大量導入・次世代電力ネットワーク小委員会"),
        ("Next Generation Power System", "次世代電力系統ワーキンググループ"),
        ("Distributed Power Systems", "次世代の分散型電力システムに関する検討会"),
        ("Watt Bit Collaboration", "ワット・ビット連携官民懇談会"),
        ("Carbon Management", "カーボンマネジメント小委員会"),
        ("Simultaneous Markets", "同時市場の在り方等に関する検討会"),
        ("Adjustment Capacity", "調整力及び需給バランス評価等に関する委員会")
    ],
    "ja": [
        ("電力・ガス基本政策小委員会", "Basic Electricity & Gas Policy"),
        ("再生可能エネルギー大量導入・次世代電力ネットワーク小委員会", "Renewable Energy & Networks"),
        ("次世代電力系統ワーキンググループ", "Next Generation Power System"),
        ("次世代の分散型電力システムに関する検討会", "Distributed Power Systems"),
        ("ワット・ビット連携官民懇談会", "Watt Bit Collaboration"),
        ("カーボンマネジメント小委員会", "Carbon Management"),
        ("同時市場の在り方等に関する検討会", "Simultaneous Markets"),
        ("調整力及び需給バランス評価等に関する委員会", "Adjustment Capacity")
    ]
}

EXAMPLE_QUESTIONS = {
    "en": [
        "What external changes are impacting Japan's electricity system?",
        "What are the main challenges in grid modernization?",
        "How is Japan addressing carbon management?",
        "What are the key renewable energy deployment strategies?",
        "How does the distributed power system work?"
    ],
    "ja": [
        "日本の電力システムに影響を与える外部変化は何ですか？",
        "電力市場の改革について教えてください",
        "日本の再生可能エネルギーの現状は？",
        "分散型電力システムの課題は何ですか？",
        "カーボンマネジメントの具体的な取り組みは？"
    ]
}

# Page configuration
st.set_page_config(
    page_title="METI Committee Information Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def get_css_for_theme(theme_mode):
    """Return CSS based on theme mode"""
    if theme_mode == 'dark':
        return """
<style>
    /* Global Styles - Dark Mode */
    .main {
        padding-top: 1rem;
    }
    
    /* Theme switcher */
    .theme-switcher {
        position: fixed;
        top: 10px;
        right: 80px;
        z-index: 999;
        background: linear-gradient(135deg, #000000 0%, #1AE315 100%);
        border-radius: 20px;
        padding: 6px 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Header - Dark */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #000000 0%, #1AE315 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(26, 227, 21, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'%3E%3Cpath d='m0 40l40-40h-40v40zm40 0v-40h-40l40 40z'/%3E%3C/g%3E%3C/svg%3E");
    }
    
    .main-header h1, .main-header p {
        position: relative;
        z-index: 1;
        margin: 0;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        
    }
    

    .main-header p {
        
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    
    /* Cards - Dark */
    .committee-card {
        background: linear-gradient(145deg, #000000 0%, #111111 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #1AE315;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        color: white;
    }
    
    .committee-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(26, 227, 21, 0.2);
        
    }
    
    .metric-card {
        background: linear-gradient(145deg, #000000 0%, #111111 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        text-align: center;
        border: 2px solid #1AE315;
        transition: all 0.3s ease;
        color: white;
    }
    
    .source-doc {
        background: linear-gradient(145deg, #000000 0%, #0a0a0a 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #1AE315;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        color: white;
    }
    
    .question-box {
        background: linear-gradient(145deg, #111111 0%, #000000 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1AE315;
        margin-bottom: 1rem;
        color: white;
    }
    
    .answer-box {
        background: linear-gradient(145deg, #000000 0%, #111111 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1AE315;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 227, 21, 0.2);
        color: white;
    }
    
    /* Input styling - Dark */
    .stTextArea textarea {
        background: linear-gradient(145deg, #000000 0%, #111111 100%) !important;
        border: 2px solid #333 !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 16px !important;
        
    }
    
    .stTextArea textarea:focus {
        border-color: #1AE315 !important;
        box-shadow: 0 0 15px rgba(26, 227, 21, 0.3) !important;
    }
    
    /* Button styling - Dark */
    button[kind="primary"] {
        background: linear-gradient(135deg, #000000 0%, #1AE315 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(26, 227, 21, 0.3) !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(26, 227, 21, 0.4) !important;
    }
    
    /* Slider styling - Dark */
    div[data-testid="stSlider"] > div > div > div {
        background: linear-gradient(135deg, #000000 0%, #1AE315 100%) !important;
        height: 6px !important;
        border-radius: 5px !important;
    }

    div[data-testid="stSlider"] > div > div > div > div {
        background-color: white !important;
        border: 2px solid #1AE315 !important;
        height: 20px !important;
        width: 20px !important;
        margin-top: -7px !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 10px rgba(26, 227, 21, 0.3) !important;
    }

    div[data-testid="stSlider"] label {
        color: white !important;
        font-weight: bold !important;
    }

    /* Force slider value styling with higher specificity - Dark */
    div[data-testid="stSlider"] * {
        --slider-value-color: white !important;
    }

    div[data-testid="stSlider"] [role="slider"]::after,
    div[data-testid="stSlider"] [role="slider"]::before {
        color: white !important;
        background-color: #000000 !important;
        border: 2px solid #1AE315 !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        font-weight: bold !important;
        font-size: 14px !important;
    }

    /* Nuclear option - target all text in slider area - Dark */
    div[data-testid="stSlider"] * {
        color: white !important;
    }

    /* Override any inline styles - Dark */
    div[data-testid="stSlider"] [style*="color"] {
        color: white !important;
        background-color: #000000 !important;
        border: 2px solid #1AE315 !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        font-weight: bold !important;
    }
</style>
"""
    else:  # light mode
        return """
<style>
    /* Global Styles - Light Mode */
    .main {
        padding-top: 1rem;
        background-color: #f8f9fa;
    }
    
    /* Theme switcher */
    .theme-switcher {
        position: fixed;
        top: 10px;
        right: 80px;
        z-index: 999;
        background: linear-gradient(135deg, #1AE315 0%, #0ea312 100%);
        border-radius: 20px;
        padding: 6px 12px;
        box-shadow: 0 4px 12px rgba(26, 227, 21, 0.3);
    }
    
    /* Header - Light */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #1AE315 100%);
        color: #333;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(26, 227, 21, 0.2);
        border: 2px solid #1AE315;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #333;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        color: #555;
    }
    
    /* Cards - Light */
    .committee-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #1AE315;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        color: #333;
        border: 1px solid #e9ecef;
    }
    
    .committee-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(26, 227, 21, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #1AE315;
        transition: all 0.3s ease;
        color: #333;
    }
    
    .source-doc {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #1AE315;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #333;
        border: 1px solid #e9ecef;
    }
    
    .question-box {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1AE315;
        margin-bottom: 1rem;
        color: #333;
        border: 1px solid #e9ecef;
    }
    
    .answer-box {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1AE315;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 227, 21, 0.1);
        color: #333 !important;
    }
    
    /* Input styling - Light */
    .stTextArea textarea {
        background: #ffffff !important;
        border: 2px solid #e9ecef !important;
        border-radius: 12px !important;
        color: #333 !important;
        font-size: 16px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #1AE315 !important;
        box-shadow: 0 0 15px rgba(26, 227, 21, 0.2) !important;
    }
    
    /* Button styling - Light */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1AE315 0%, #0ea312 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(26, 227, 21, 0.3) !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(26, 227, 21, 0.4) !important;
    }
    
    /* Secondary buttons - Light */
    .stButton > button:not([kind="primary"]) {
        background: #ffffff !important;
        color: #1AE315 !important;
        border: 2px solid #1AE315 !important;
        border-radius: 8px !important;
    }
    
    /* Slider styling - Light */
    div[data-testid="stSlider"] > div > div > div {
        background: linear-gradient(135deg, #1AE315 0%, #0ea312 100%) !important;
        height: 6px !important;
        border-radius: 5px !important;
    }

    div[data-testid="stSlider"] > div > div > div > div {
        background-color: white !important;
        border: 2px solid #1AE315 !important;
        height: 20px !important;
        width: 20px !important;
        margin-top: -7px !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 10px rgba(26, 227, 21, 0.3) !important;
    }

    div[data-testid="stSlider"] label {
        color: #333 !important;
        font-weight: bold !important;
    }

    /* Force slider value styling - Light */
    div[data-testid="stSlider"] * {
        color: #333 !important;
    }

    div[data-testid="stSlider"] [style*="color"] {
        color: #333 !important;
        background-color: #ffffff !important;
        border: 2px solid #1AE315 !important;
        border-radius: 6px !important;
        padding: 6px 10px !important;
        font-weight: bold !important;
        box-shadow: 0 2px 8px rgba(26, 227, 21, 0.2) !important;
    }
    
    /* Selectbox styling - Light */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 2px solid #e9ecef !important;
        border-radius: 8px !important;
        color: #333 !important;
    }
    
    /* Text color fixes for light mode - Enhanced */
    .stMarkdown, .stMarkdown * {
        color: #333 !important;
    }

    /* Main content area text */
    .main .stMarkdown p, 
    .main .stMarkdown h1, 
    .main .stMarkdown h2, 
    .main .stMarkdown h3,
    .main .stMarkdown span,
    .main .stMarkdown div,
    .main p, .main h1, .main h2, .main h3, .main span, .main div {
        color: #333 !important;
    }

    /* Question and answer sections */
    [data-testid="stVerticalBlock"] *,
    [data-testid="element-container"] *,
    .element-container * {
        color: #333 !important;
    }

    /* Headers and subheaders */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    header, .stHeader * {
        color: #333 !important;
    }

    /* Text areas and form labels */
    .stTextArea label, 
    .stTextInput label,
    .stSelectbox label,
    .stSlider label {
        color: #333 !important;
    }

    /* Metrics and status text */
    [data-testid="metric-container"] *,
    [data-testid="metric-container"] .metric-label,
    [data-testid="metric-container"] .metric-value {
       color: #333 !important;
    }

   /* Nuclear option - Force all text to be dark except buttons */
    .stApp * {
       color: #333 !important;
    }

    /* Keep button text white and green text where needed */
    button[kind="primary"], button[kind="primary"] * {
       color: white !important;
    }

    .stButton > button:not([kind="primary"]) {
       color: #1AE315 !important;
    }
    
    /* Keep green accent colors */
    .committee-card small[style*="color: #1AE315"] {
       color: #1AE315 !important;
    }
    
    /* Sidebar styling - Light */
    .css-1d391kg {
        background: #f8f9fa !important;
    }
    
    /* Metric styling - Light */
    [data-testid="metric-container"] {
        background: #ffffff !important;
        border: 2px solid #1AE315 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(26, 227, 21, 0.1) !important;
        color: #333 !important;
    }
</style>
"""

# Render theme and language switchers
def render_switchers():
    """Render the theme and language switchers in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        
        # Theme switcher
        st.markdown(f"### {get_text('theme_mode')}")
        theme_mode = st.radio(
            get_text("theme_mode"),
            ["dark", "light"],
            format_func=lambda x: get_text("dark_mode") if x == "dark" else get_text("light_mode"),
            index=0 if st.session_state.theme_mode == "dark" else 1,
            key="theme_switcher",
            horizontal=True
        )
        
        if theme_mode != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_mode
            st.rerun()
        
        st.markdown("---")
        
        # Language switcher
        current_lang = st.radio(
            "🌐 Language / 言語:",
            ["en", "ja"],
            format_func=lambda x: "🇺🇸 English" if x == "en" else "🇯🇵 日本語",
            index=0 if st.session_state.language == "en" else 1,
            key="lang_switcher",
            horizontal=True
        )
        
        if current_lang != st.session_state.language:
            st.session_state.language = current_lang
            st.rerun()

def get_text(key):
    """Get text in current language"""
    return LANGUAGES[st.session_state.language].get(key, key)

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
        
        with st.spinner(get_text("searching")):
            result = qa_chain.invoke({"query": question})
        
        return result
    
    except Exception as e:
        st.error(f"Error querying system: {str(e)}")
        return None

def main():
    # Apply theme-based CSS
    st.markdown(get_css_for_theme(st.session_state.theme_mode), unsafe_allow_html=True)
    
    # Render switchers
    render_switchers()
    
    # Header
    st.markdown(f"""
    <div class="main-header fade-in">
        <h1>🏛️ {get_text("title")}</h1>
        <p>{get_text("subtitle")} {get_text("coming_soon")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.system_initialized:
        with st.spinner(get_text("initializing")):
            st.session_state.rag_system = initialize_rag_system()
            if st.session_state.rag_system:
                st.session_state.system_initialized = True
                st.success(get_text("system_initialized"))
            else:
                st.error(get_text("system_failed"))
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header(get_text("config_header"))
        
        # Prompt type selection
        prompt_type = st.selectbox(
            get_text("response_style"),
            ["comprehensive", "simple"],
            format_func=lambda x: get_text("comprehensive") if x == "comprehensive" else get_text("simple"),
            help=get_text("comprehensive_help")
        )
        
        # Number of documents to retrieve
        retrieval_k = st.slider(
            get_text("documents_retrieve"),
            min_value=1,
            max_value=10,
            value=5,
            help=get_text("retrieval_help")
        )
        
        st.markdown("---")
        
        # Committee information
        st.header(get_text("committee_coverage"))
        
        committees = COMMITTEES_INFO[st.session_state.language]
        
        for name_primary, name_secondary in committees:
            st.markdown(f"""
            <div class="committee-card">
                <strong>{name_primary}</strong><br>
                <small style="color: #1AE315;">{name_secondary}</small><br>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Language info
        st.header(get_text("language_support"))
        st.info(get_text("language_support_text"))
        
        # Example questions
        st.header(get_text("example_questions"))
        example_questions = EXAMPLE_QUESTIONS[st.session_state.language]
        
        for i, question in enumerate(example_questions):
            if st.button(f"💡 {question[:35]}...", key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Main content area
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.markdown(f'<div class="fade-in">', unsafe_allow_html=True)
        st.header(get_text("ask_question"))
        
        # Question input
        if 'current_question' in st.session_state:
            question = st.text_area(
                f"{get_text('ask_question')}:",
                value=st.session_state.current_question,
                height=120,
                placeholder=get_text("question_placeholder"),
                key="question_input"
            )
            del st.session_state.current_question
        else:
            question = st.text_area(
                f"{get_text('ask_question')}:",
                height=120,
                placeholder=get_text("question_placeholder"),
                key="question_input_default"
            )
        
        # Query button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button(get_text("search_button"), type="primary", use_container_width=True):
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
                            'retrieval_k': retrieval_k,
                            'language': st.session_state.language
                        })
                        
                        st.success(get_text("query_success"))
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(get_text("query_failed"))
                else:
                    st.warning(get_text("enter_question"))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="fade-in">', unsafe_allow_html=True)
        st.header(get_text("system_status"))
        
        # System metrics
        if st.session_state.rag_system:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{get_text("system_online")}</h3>
                <p>{get_text("system_operational")}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Query statistics
        total_queries = len(st.session_state.chat_history)
        st.metric(get_text("total_queries"), total_queries)
        
        if st.session_state.chat_history:
            latest_query = st.session_state.chat_history[-1]['timestamp']
            st.metric(get_text("last_query"), latest_query)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown(f'<div class="fade-in">', unsafe_allow_html=True)
        st.header(get_text("query_results"))
        
        # Display latest result
        latest = st.session_state.chat_history[-1]
        
        # Question
        st.subheader(get_text("question_label"))
        st.markdown(f"""
        <div class="question-box">
            {latest['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Answer
        st.subheader(get_text("answer_label"))
        st.markdown(f"""
        <div class="answer-box">
            {latest['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        # Source documents
        if latest['source_documents']:
            st.subheader(get_text("source_documents"))
            
            for i, doc in enumerate(latest['source_documents']):
                
                #Extract metadata for link creation
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                s3_uri = metadata.get('x-amz-bedrock-kb-source-uri', metadata.get('source', ''))
                page_number = metadata.get('x-amz-bedrock-kb-page-number')
                
                #Create PDF link
                if s3_uri and s3_uri.startswith("s3://"):
                    pdf_link = create_presigned_pdf_link(s3_uri, page_number)
                    filename = s3_uri.split("/")[-1]
                else:
                    pdf_link = "📄 Source document"
                    filename = f"Document {i+1}"
                    
                with st.expander(f"📄 {filename}", expanded=False):
                    # Show PDF link prominently
                    if s3_uri and s3_uri.startswith('s3://'):
                        st.markdown(f"**🔗 View PDF:** {pdf_link}")
                        st.markdown("---")
            
                    st.markdown(f"""
                    <div class="source-doc">
                       <strong>{get_text("content_preview")}</strong><br><br>
                       {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
                    # Show metadata in a collapsible section
                    if metadata:
                         with st.expander("📋 Document Metadata", expanded=False):
                               st.json(metadata)
                 
        
        # Query details
        with st.expander(f"🔍 {get_text('query_details')}", expanded=False):
            col_detail1, col_detail2, col_detail3 = st.columns(3)
            with col_detail1:
                st.write(f"**{get_text('prompt_type')}** {latest['prompt_type']}")
            with col_detail2:
                st.write(f"**{get_text('documents_retrieved')}** {latest['retrieval_k']}")
            with col_detail3:
                st.write(f"**{get_text('timestamp')}** {latest['timestamp']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat history
    if len(st.session_state.chat_history) > 1:
        st.markdown("---")
        st.markdown(f'<div class="fade-in">', unsafe_allow_html=True)
        st.header(get_text("query_history"))
        
        # Create tabs for better organization
        if len(st.session_state.chat_history) > 3:
            # Show recent queries in a more compact format
            for i, entry in enumerate(reversed(st.session_state.chat_history[:-1])):
                query_num = len(st.session_state.chat_history) - i - 1
                with st.expander(f"🔍 Query {query_num}: {entry['question'][:60]}...", expanded=False):
                    col_hist1, col_hist2 = st.columns([3, 1])
                    with col_hist1:
                        st.write(f"**{get_text('question_label')}** {entry['question']}")
                        st.write(f"**{get_text('answer_label')}**")
                        st.write(entry['answer'][:300] + "..." if len(entry['answer']) > 300 else entry['answer'])
                    with col_hist2:
                        st.write(f"**{get_text('timestamp')}**")
                        st.write(entry['timestamp'])
                        st.write(f"**Language:** {'🇺🇸 EN' if entry.get('language', 'en') == 'en' else '🇯🇵 JA'}")
        else:
            # Show all queries in detail
            for i, entry in enumerate(reversed(st.session_state.chat_history[:-1])):
                query_num = len(st.session_state.chat_history) - i - 1
                with st.expander(f"Query {query_num}: {entry['question'][:50]}...", expanded=False):
                    st.write(f"**{get_text('question_label')}** {entry['question']}")
                    st.write(f"**{get_text('answer_label')}** {entry['answer']}")
                    st.write(f"**{get_text('timestamp')}** {entry['timestamp']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear history button
    if st.session_state.chat_history:
        st.markdown("---")
        col_clear1, col_clear2, col_clear3 = st.columns([1, 1, 1])
        with col_clear2:
            if st.button(get_text("clear_history"), use_container_width=True):
                st.session_state.chat_history = []
                st.success("✅ History cleared!" if st.session_state.language == "en" else "✅ 履歴をクリアしました！")
                time.sleep(1)
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; color: #{'666' if st.session_state.theme_mode == 'dark' else '555'}; font-size: 0.9rem;">
        <p>⚡ Powered by RAG Technology | 🏛️ Official METI Committee Documents | 🤖 Claude AI | Agile Energy X ™️ | Created by Arghadeep Biswas</p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
            {"This system provides information from official METI committee meetings. For the most current information, please refer to official METI publications." if st.session_state.language == "en" else "このシステムは公式のMETI委員会会議の情報を提供します。最新の情報については、公式のMETI出版物をご参照ください。"}
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()