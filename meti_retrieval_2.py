import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# AWS + Pinecone Configs
AWS_REGION = os.getenv("AWS_REGION")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

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

# Define comprehensive custom prompt templates
SYSTEM_PROMPT_TEMPLATE = """You are an expert assistant specializing in Japan's electricity and energy policy, with access to detailed information from 8 key METI (Ministry of Economy, Trade and Industry) committee meetings held in 2025.

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

## Response Guidelines
When answering questions, follow this structure:

1. **Direct Answer**: Provide a clear, concise answer to the question
2. **Supporting Details**: Include relevant details, data, or policy specifics from the meetings
3. **Committee Context**: Explain which committee(s) discussed this topic and why it's relevant
4. **Source Attribution**: Clearly cite the specific meeting(s) and documents used
5. **Related Information**: Mention any related topics or cross-committee discussions when relevant

## Response Requirements
- ✅ Cite specific committee meetings and sources
- ✅ Provide context about which committee discussed the topic
- ✅ Use official METI terminology and policy language
- ✅ Include Japanese terms when appropriate for authenticity
- ✅ Be accurate and precise, avoiding speculation beyond documented information
- ✅ If information is not available in the documents, clearly state this limitation
- ✅ Maintain professional and authoritative tone appropriate for government policy discussions
- ✅ **LANGUAGE MATCHING: If the user asks a question in Japanese, respond entirely in Japanese. If the user asks in English, respond in English. Match the user's language preference.**

## Context Information
Based on the following retrieved documents from METI committee meetings:

{context}

## User Question
{question}

## Response
Please provide a comprehensive answer following the guidelines above:"""

# Alternative shorter prompt for simple queries
SIMPLE_PROMPT_TEMPLATE = """You are a METI energy policy expert with access to 2025 committee meeting documents from 8 key committees covering electricity, renewable energy, grid systems, and market reforms.

Context from retrieved documents:
{context}

Question: {question}

Instructions:
- Answer based solely on the provided context
- Cite specific committee meetings when possible
- Include Japanese terms for key concepts
- If information is not in the context, state this clearly
- Maintain professional tone
- **IMPORTANT: If the user's {question} is in Japanese, respond entirely in Japanese. If in English, respond in English.**
Answer:"""

# Create multiple prompt options
def create_prompt_template(template_type="comprehensive"):
    """Create different types of prompt templates"""
    
    if template_type == "comprehensive":
        template = SYSTEM_PROMPT_TEMPLATE
    elif template_type == "simple":
        template = SIMPLE_PROMPT_TEMPLATE
    else:
        raise ValueError("template_type must be 'comprehensive' or 'simple'")
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Initialize with comprehensive prompt (default)
prompt = create_prompt_template("comprehensive")

# The QA chain will be created dynamically in the query functions
# This allows for flexible prompt template selection

# Enhanced function to query the system with different prompt options
def query_meti_committees(question, prompt_type="comprehensive", retrieval_k=5):
    """
    Query the METI committee knowledge base with customizable prompts
    
    Args:
        question (str): The user's question
        prompt_type (str): "comprehensive" or "simple" prompt template
        retrieval_k (int): Number of documents to retrieve (default: 5)
    
    Returns:
        dict: Query results with answer and source documents
    """
    try:
        # Create the appropriate prompt template
        current_prompt = create_prompt_template(prompt_type)
        
        # Create QA chain with the selected prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": retrieval_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": current_prompt}
        )
        
        # Execute the query
        result = qa_chain.invoke({"query": question})
        
        # Display results
        print(f"\n🔍 Query: {question}")
        print(f"📝 Prompt Type: {prompt_type}")
        print(f"📄 Documents Retrieved: {len(result['source_documents'])}")
        print(f"\n📢 Answer:\n{result['result']}")
        
        if result['source_documents']:
            print(f"\n Source Documents:")
            for i, doc in enumerate(result['source_documents']):
                print(f"\n--- Document {i+1} ---")
                print(f"Content Preview: {doc.page_content[:300]}...")
                if doc.metadata:
                    print(f"Metadata: {doc.metadata}")
        
        return result
    
    except Exception as e:
        print(f"❌ Error querying the system: {str(e)}")
        return None

# Interactive query function
def interactive_query():
    """Interactive mode for querying the system"""
    print("🏛️  METI Committee Information Assistant")
    print("=" * 50)
    print("Available prompt types: 'comprehensive' (default) or 'simple'")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        question = input("\n💬 Enter your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            print("❓ Please enter a question.")
            continue
        
        # Ask for prompt type
        prompt_choice = input("🎯 Prompt type (comprehensive/simple) [default: comprehensive]: ").strip().lower()
        if prompt_choice not in ['comprehensive', 'simple']:
            prompt_choice = 'comprehensive'
        
        # Ask for number of documents to retrieve
        try:
            k_docs = input("📄 Number of documents to retrieve [default: 5]: ").strip()
            k_docs = int(k_docs) if k_docs else 5
        except ValueError:
            k_docs = 5
        
        # Execute query
        result = query_meti_committees(question, prompt_choice, k_docs)
        
        if result:
            print("\n" + "="*100)
            continue_prompt = input("\n🔄 Continue with another question? (y/n): ").strip().lower()
            if continue_prompt in ['n', 'no']:
                print("👋 Goodbye!")
                break
        else:
            print("❌ Query failed. Please try again.")

# Batch query function for testing
def batch_query_test(test_queries, prompt_type="comprehensive"):
    """Run multiple test queries in batch"""
    print(f"\n🧪 Running batch test with {prompt_type} prompt")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔬 Test Query {i}/{len(test_queries)}")
        result = query_meti_committees(query, prompt_type)
        
        if i < len(test_queries):
            print("\n" + "="*100)
            input("⏸️  Press Enter to continue to next query...")
    
    print("\n✅ Batch testing completed!")

# Example usage
if __name__ == "__main__":
    # Test queries covering different committee topics
    test_queries = [
        "What external changes are impacting Japan's electricity system?",
        "What are the key challenges in renewable energy integration discussed in the committees?",
        "What is the current status of electricity market reforms in Japan?",
        "How is Japan addressing carbon management in the electricity sector?",
        "What are the main focus areas of the Watt Bit Collaboration Forum?",
        "What distributed power system technologies are being discussed?"
    ]
    
    # Option 1: Interactive mode
    # interactive_query()
    
    # Option 2: Batch testing mode
    batch_query_test(test_queries, "comprehensive")
    
    # Option 3: Single query example
    # single_result = query_meti_committees(
    #     "What external changes are impacting Japan's electricity system?",
    #     prompt_type="comprehensive",
    #     retrieval_k=3
    # )