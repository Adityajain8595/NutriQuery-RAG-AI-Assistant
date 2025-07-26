import os
from dotenv import load_dotenv
from pathlib import Path
import pdfplumber
import subprocess
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from icmr_nin import load_embeddings
from session_store import get_session_history
from langchain_core.runnables import Runnable
from langchain_core.documents import Document

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("nutriquery-rag-hybrid-search")

def process_query(user_input: str, session_id: str):
    embeddings = load_embeddings()
    bm25_encoder = BM25Encoder().default()
    pc_retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=os.environ["GROQ_API_KEY"], temperature=0.0)

    class ThresholdedHybridRetriever(Runnable):
        def __init__(self, retriever, threshold=0.6, k=10):
            self.retriever = retriever
            self.threshold = threshold
            self.k = k

        def invoke(self, query: str, config=None) -> list[Document]:
            # ✅ Embed dense and sparse representations
            dense_vector = self.retriever.embeddings.embed_query(query)
            sparse_vector = self.retriever.sparse_encoder.encode_queries([query])[0]  # Not encode_documents!

            # ✅ Perform hybrid search using Pinecone index
            results = self.retriever.index.query(
                vector=dense_vector,
                sparse_vector=sparse_vector,
                top_k=self.k,
                include_metadata=True,
                namespace=self.retriever.namespace,
            )

            # ✅ Filter matches using similarity threshold
            filtered_matches = [match for match in results["matches"] if match["score"] >= self.threshold]

            # ✅ Convert to LangChain Documents
            return [
                Document(page_content=match["metadata"].get("text", ""), metadata=match["metadata"])
                for match in filtered_matches
            ]
        
    threshold_retriever = ThresholdedHybridRetriever(pc_retriever, threshold=0.6, k=10)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, provide context for the question."
                "Reformulate as standalone question if needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, threshold_retriever, contextualize_prompt)

    system_prompt = (
        """
        You are NutriQuery, a knowledgeable nutrition assistant **only nutrition-related questions** strictly based on the provided ICMR-NIN documents.
        If a question is unrelated to nutrition/health or whose answer is not present in the provided documents, after close examination of the context and the user's question,
        Say: **"⚠️ I'm here to assist with nutrition-related questions only. The information you've requested isn't aligned with my objectives to help you. Feel free to ask anything related to diet, health, or nutrition!"**

        Always answer clearly, structurally, and in detail, by sticking to the facts, reports, and statistics in the provided context.
        Sound just like a human nutritionist would do.
        1. Start with a short personalized overview of the user's situation/problem.
        2. Then organize complete info with bullets or sections.
        3. Suggest 2-3 follow-up questions at the end.

        Do not mention 'context' or 'provided text'. Show confidence in your answers.

        <context>
        {context}
        </context>
        """
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        return_source_documents=True
    )

    result = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    chat = get_session_history(session_id)
    readable_history = [
        f"{msg.content}" if msg.type == "human" else f"{msg.content}"
        for msg in chat.messages
    ]
    # source_docs = result.get("context", [])  # Retrieved documents
    # print(source_docs)
    return result["answer"], readable_history

def add_pdf_to_retriever(uploaded_file):
    embeddings = load_embeddings()
    bm25_encoder = BM25Encoder().default()
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)
    temp_path = Path("temp.pdf")
    file_name = uploaded_file.filename
    temp_path.write_bytes(uploaded_file.file.read())
    ocr_path = os.path.join(temp_path, "ocr_temp")
    
    chunks = []

    try:
        # Run OCR to ensure searchable text
        subprocess.run(
            ["ocrmypdf", "--force-ocr", str(temp_path), ocr_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Use pdfplumber to extract clean text
        raw_text = ""
        with pdfplumber.open(ocr_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    raw_text += text + "\n\n"

        if not raw_text.strip():
            print(f"⚠️ Empty text after OCR: {temp_path.name}")
            return False

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents(
            [raw_text],
            metadatas=[{"source": file_name}]
        )

    except subprocess.CalledProcessError as e:
        print(f"❌ OCR failed for {file_name}: {e.stderr.decode()}")
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
    finally:
        temp_path.unlink(missing_ok=True)
        if os.path.exists(ocr_path):
            os.remove(ocr_path)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    for meta in metadatas:
        meta["source_institute"] = "User-Uploaded Document"
    retriever.add_texts(texts, metadatas=metadatas)
    return True
