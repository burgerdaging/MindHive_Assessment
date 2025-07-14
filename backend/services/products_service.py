from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import settings
import logging
import asyncio # Keep this import, as ainvoke is an async operation

logger = logging.getLogger(__name__)

class ZUSProductsService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=settings.GEMINI_API_KEY,
            temperature=0.0
        )

        # MongoDB setup
        self.client = MongoClient(settings.MONGODB_ATLAS_CLUSTER_URI)
        # Ensure this collection name is correct for your product data
        self.collection = self.client[settings.DB_NAME][settings.ZUS_COLLECTION_NAME]
        # Ensure this index name is correct for your product data vector search
        self.vector_index_name = settings.ATLAS_VECTOR_SEARCH_INDEX_NAME

        # Initialize vector store and retriever
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=self.vector_index_name,
            relevance_score_fn="cosine",
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  # Default number of results
        )

        # Setup prompt template
        self.system_prompt = """
        You are a ZUS Coffee product expert. Use the given context to answer the question.
        If you don't know the answer, say you don't know but suggest similar products.
        Include specific details like prices and colors when available.
        Keep the answer concise but helpful.
        
        Context: {context}
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])

        # Create chains
        self.document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt
        )

        self.retrieval_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=self.document_chain
        )

    async def search_products(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search ZUS products using vector store"""
        try:
            logger.info(f"Searching ZUS products for: {query}")

            # Update retriever with requested top_k
            self.retriever.search_kwargs["k"] = top_k

            # Use the retrieval chain with the actual user query
            response = await self.retrieval_chain.ainvoke({"input": query})

            # Format results
            vector_results = []
            if response.get("context"):
                for doc in response["context"]:
                    vector_results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "ZUS Document"),
                        "type": "stored_document",
                        "metadata": doc.metadata
                    })

            final_answer = response.get("answer", "").strip()

            # Determine success based on whether a meaningful answer was generated
            # and if it doesn't contain a "don't know" phrase (from your system prompt)
            is_successful = bool(final_answer) and "don't know" not in final_answer.lower()

            # If no meaningful answer, generate a default message
            if not is_successful:
                final_answer = self._generate_no_results_message(query)

            return {
                "query": query,
                "vector_results": vector_results,
                "live_search_results": [],
                "combined_summary": final_answer,
                "success": is_successful,
                "sources": ["ZUS Document Store"] if vector_results else []
            }

        except Exception as e:
            logger.error(f"Error in product search: {e}", exc_info=True)
            return {
                "query": query,
                "vector_results": [],
                "live_search_results": [],
                "combined_summary": self._generate_no_results_message(query),
                "success": False,
                "sources": []
            }

    def _generate_no_results_message(self, query: str) -> str:
        """Generate helpful message when no results are found"""
        return f"""I couldn't find specific information about "{query}" in our current ZUS Coffee product database.

**Suggestions:**
- Try different search terms like "tumbler", "mug", or "coffee cup"
- Visit shop.zuscoffee.com for current product listings
- Check these popular ZUS Coffee drinkware categories:
  - Tumblers (600ml, 500ml)
  - Travel mugs
  - Cold cups (650ml)
"""