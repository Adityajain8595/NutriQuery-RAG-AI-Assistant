from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

session_histories = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = ChatMessageHistory()
    history = session_histories[session_id]
    if len(history.messages) > 10:
        history.messages = history.messages[-10:]
    return history