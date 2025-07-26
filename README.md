# 🥦 NutriQuery - AI-Powered Nutrition Assistant

NutriQuery is a powerful AI-based Retrieval-Augmented Generation (RAG) assistant designed to provide reliable and contextual answers to nutrition-related queries. It leverages the ICMR-NIN official documents powered with AI, combining dense and sparse (hybrid) search using Pinecone, OCR-based PDF ingestion, and conversational memory to emulate a helpful human-like nutritionist.

---

## 🚀 Features

🔍 Hybrid Search (Dense + BM25) for accurate context retrieval from ICMR-NIN PDFs.

🧠 Conversational Memory to maintain chat history across sessions.

🗣️ Voice Input for hands-free interaction.

📄 PDF Upload for personalized document analysis.

🎯 Nutrition-only Focus with filtered responses based on authoritative sources.

💡 Smart Follow-ups that promote interactive learning and engagement.

---

📸 Demo Video:





---

## 🧠 How It Works

Pinecone index is created from Pinecone console on web through its API. Then, PineconeHybridSearchRetriever is initialized in globals.py file.

1. Document Ingestion:

All ICMR-NIN PDFs are OCR-processed using ocrmypdf to ensure text extraction, through icmr_nin.py.

Texts are chunked and embedded using HuggingFaceEmbeddings + BM25Encoder.

Stored in Pinecone with hybrid search indexing.

**User Query Flow:**

User asks a question (via text or voice).

NutriQuery performs hybrid retrieval from the vector DB.

An LLM (gemma2-9b-it via Groq) answers using retrieved context, maintaining chat memory, through rag_chain.py and session_store.py.

2. PDF Upload:

Users can upload their own nutrition-related PDFs.

Text is extracted, chunked, embedded, and added to the vector index on-the-fly.

3. Frontend:

A stylish Flask-powered frontend with:

Input box + mic support

Real-time responses with markdown rendering

Sidebar for chat history containing chats with their timestamps. 

Upload interface for PDFs

Feature cards and disclaimer

---

## 🛠️ Tech Stack

1. LLM: gemma2-9b-it via Groq API
2. Embeddings: all-MiniLM-L6-v2 (dense) + BM25 (sparse)
3. Vector DB: Pinecone Hybrid Search
4. OCR:	ocrmypdf + pdfplumber
5. Backend: FastAPI
6. Frontend: Flask + HTML/CSS/JavaScript
7. Prompt Chains: LangChain's history-aware retriever, document chain

---

## ⚙️ Setup Instructions

1. Environment Setup

> git clone https://github.com/Adityajain8595/NutriQuery-RAG-AI-Assistant.git

> cd NutriQuery-RAG-AI-Assistant

> conda create -n venv python=3.10

> conda activate venv/


2. Environment Variables

Create a .env file with the following:

> LANGCHAIN_API_KEY=your_langchain_api_key

> HF_TOKEN=your_huggingface_token

> GROQ_API_KEY=your_groq_api_key

> PINECONE_API_KEY=your_pinecone_api_key


3. Embed ICMR-NIN PDFs

Placed PDFs inside: backend/icmr_nin/

Then run:

> cd backend/                      -> goes to backend folder

> python global.py                 -> creates index if not exists and initializes retriever from that

> python icmr_nin.py               -> OCR, chunk, embed in batches, and push them to Pinecone 

 
4. Start Backend API

> cd backend/

> uvicorn app:app --reload --port 8000   (port 8000 taken as an example)


5. Start Frontend

> cd frontend                      -> goes to frontend folder

> python app.py                    -> runs Flask app on specified port

Navigate to http://localhost:5000 (or any host) in your browser.

---

## 🧪 API Endpoints

1. /	       GET	     App's health check
2. /ask	       POST	     Ask a question (query, session_id)
3. /upload	   POST	     Upload PDF file (file)

---

## 🧠 Potential Applications:

🏥 Assist doctors, dietitians, and nutritionists in quickly extracting dietary guidelines and nutrient data from ICMR documents during consultations.

📚 Support students and researchers in querying government-backed nutrition data for assignments, publications, and projects.

🧑‍🍳 Help diet coaches access accurate food composition values and dietary allowances when crafting personalized diet plans.

📱 Embed NutriQuery’s API to enable nutrition search and recommendation features powered by reliable Indian dietary data.

(Still, there is much work to done! Looking to make NutriQuery more intelligent with addition of WHO and ICMR guidelines and reports.)

***NOTE:** This application is made with the aim of assisting people with their general queries about nutrition and maintaing proper health. For emergency and medical help, please consult a healthcare expert for personalized guidance.

---

## 🤝 Author

First of all, credits to ICMR-NIN, my source of scientific nutrition documents, for their incredible work in nutrition space.

Made with love by me, Aditya Jain. Would love to connect with you and hear your feedback!

Connect with me on 📧 LinkedIn: https://www.linkedin.com/in/adityajain8595/


