from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import pdfplumber
import subprocess
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tqdm import tqdm
from more_itertools import chunked 
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
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
def load_embeddings():
    return HuggingFaceEndpointEmbeddings(
        api_key = os.getenv("HF_TOKEN"),
        model_name="all-MiniLM-L6-v2",
    )

index = pc.Index(index_name)

embeddings = load_embeddings()

BM25_Encoder = BM25Encoder().default()

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=BM25_Encoder, index=index)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

if index.describe_index_stats().total_vector_count > 0:
    print("✅ ICMR-NIN embeddings already exist. Skipping embedding.")
else:
    # 🧠 Embed logic for ICMR-NIN documents
    folder_path= r"C:\Users\ADMIN\OneDrive\Documents\Code\Langchain_Projects\NutriQuery-RAG-Chatbot\backend\icmr_nin"
    ocr_output_dir = os.path.join(folder_path, "ocr_temp")
    os.makedirs(ocr_output_dir, exist_ok=True)

    all_chunks = []
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    for pdf_file_path in pdf_files:
        print(f"Processing: {pdf_file_path}")
        ocr_pdf_path = os.path.join(ocr_output_dir, pdf_file_path.name)

        try:
            # Run OCR to ensure searchable text
            subprocess.run(
                ["ocrmypdf", "--force-ocr", str(pdf_file_path), ocr_pdf_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Use pdfplumber to extract clean text
            raw_text = ""
            with pdfplumber.open(ocr_pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        raw_text += text + "\n\n"

            if not raw_text.strip():
                print(f"⚠️ Empty text after OCR: {pdf_file_path.name}")
                continue

            # Chunking
            chunks = splitter.create_documents(
                [raw_text],
                metadatas=[{"source": pdf_file_path.name}]
            )
            all_chunks.extend(chunks)

            print(f"✅ Chunks created: {len(chunks)} for {pdf_file_path.name}")

        except subprocess.CalledProcessError as e:
            print(f"❌ OCR failed for {pdf_file_path.name}: {e.stderr.decode()}")
        except Exception as e:
            print(f"❌ Error processing {pdf_file_path.name}: {e}")

    # Extract texts and metadata separately
    texts = [doc.page_content for doc in all_chunks]
    metadatas = [doc.metadata for doc in all_chunks]

    for meta in metadatas:
        meta["source_institute"] = "ICMR-National Institute of Nutrition, India"

    batch_size = 100
    text_batches = list(chunked(texts, batch_size))
    metadata_batches = list(chunked(metadatas, batch_size))

    # Now insert with progress bar
    for i, (text_batch, meta_batch) in enumerate(tqdm(zip(text_batches, metadata_batches), total=len(text_batches), desc="Embedding Batches")):
        retriever.add_texts(text_batch, metadatas=meta_batch)

    print("✅ Chunks embedded in the vectorstore!")