from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

index_name = "nutriquery-rag-hybrid-search"
pc = Pinecone(api_key=pinecone_api_key)

"""
Use when the index is not created yet.

if index_name not in pc.list_indexes():
    print(f"🆕 Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        metric="dotproduct",
        dimension=384,
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    
"""
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)
index = pc.Index(index_name)
BM25_Encoder = BM25Encoder()
pc_retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=BM25_Encoder, index=index)