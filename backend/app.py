from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rag_chain import process_query, add_pdf_to_retriever
from session_store import session_histories

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(
    query: str = Form(...),
    session_id: str = Form(default="default_session")
):
    answer, chat = process_query(query, session_id)
    return {"answer": answer, "chat_history": chat}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        success = add_pdf_to_retriever(file)
        return JSONResponse(content={"message": "PDF added to retriever."}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)