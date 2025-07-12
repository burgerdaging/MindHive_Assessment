import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from pymongo import MongoClient 
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings



load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))


@tool
def get_current_time(query: str) -> str:
    """Returns the current time. Use this tool when the user asks for the current time."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression. Use this for any math questions."""
    try:
        return str(eval(expression)) # Be careful with eval in production, but fine for assessment
    except Exception as e:
        return f"Error evaluating expression: {e}"
    
tools = [get_current_time, calculator]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. You have access to tools to answer user questions."),
    MessagesPlaceholder(variable_name="chat_history"), # For memory
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's internal thoughts
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)



# Test 1: A simple query that doesn't need a tool
# print(agent_executor.invoke({"input": "Hello, how are you?", "chat_history": []})["output"])

# # Test 2: A query that uses the tool
# print(agent_executor.invoke({"input": "What time is it right now?", "chat_history": []})["output"])