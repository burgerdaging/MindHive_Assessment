import os 
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient 
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv


#this file used to add documents into the vector store.
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))

DB_NAME = "MongoDB"
COLLECTION_NAME = "Facts-txt"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "MindHive-Assessment"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

text_splitter = CharacterTextSplitter(separator = "\n", chunk_size=1000, chunk_overlap=200)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

vector_store.add_documents(docs)
print("Documents Added")
client.close()