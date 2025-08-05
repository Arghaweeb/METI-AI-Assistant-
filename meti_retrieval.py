import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain_aws import ChatBedrock

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
    text_key="text",  # this should match what you used during upload
    namespace=os.getenv("PINECONE_NAMESPACE")  # specify namespace
)

# Initialize LLM
llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name=AWS_REGION
)

# Create RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Run a test query
query = "What external changes are impacting Japan's electricity system?"
result = qa.invoke({"query": query})

# Print the result
print("\n📢 Answer:\n", result["result"])
print("\n📚 Source Documents:")
for i, doc in enumerate(result["source_documents"]):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")