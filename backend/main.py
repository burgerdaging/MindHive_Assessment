# backend/app.py
# Part 1 endpoint is here
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.0)

# Conversation setup
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful Mathematics teacher, Answer all questions to the best of your ability in English"
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "human",
        "{input}"
    )
])

chain = prompt | llm
store = {}

def get_message_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Pydantic model for request
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

@app.get("/")
async def home():
    return {"message": "MindHive Assessment Backend"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chain_with_history.invoke(
            {"input": request.message},
            {"configurable": {"session_id": request.session_id}}
        )
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))