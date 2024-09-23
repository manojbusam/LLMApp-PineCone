"""
FastAPI Microservice for Advanced Querying of Pinecone Index using LangChain

Author: Manoj Busam
Date: April 22, 2024

This program creates a FastAPI-based microservice that allows users to query
a previously indexed Pinecone database using advanced LangChain components.

Features:
1. FastAPI web server setup
2. LangChain integration for querying Pinecone index
3. OpenAI embeddings and LLM for query processing and response generation
4. Custom prompts and output parsers for structured responses
5. Error handling and logging

To use this microservice:
1. Install dependencies: pip install fastapi uvicorn langchain pinecone-client openai pydantic
2. Set up your Pinecone and OpenAI API keys as environment variables
3. Run the server: uvicorn main:app --reload
4. Access the API documentation at http://localhost:8000/docs
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
import pinecone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Advanced LangChain Pinecone Query Microservice", version="1.0.0")

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI Embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Initialize Pinecone vector store
vectorstore = Pinecone.from_existing_index(index_name, embeddings)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class AnalyzedResult(BaseModel):
    summary: str = Field(description="A brief summary of the retrieved information")
    key_points: list[str] = Field(description="List of key points extracted from the retrieved documents")
    relevance_score: float = Field(description="A score from 0 to 1 indicating the overall relevance of the results to the query")

class QueryResponse(BaseModel):
    query: str
    analyzed_result: AnalyzedResult
    raw_results: list

# Create a parser based on the AnalyzedResult model
parser = PydanticOutputParser(pydantic_object=AnalyzedResult)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant tasked with analyzing search results.
    Given the following query and search results, provide a summary, extract key points, and assess the overall relevance.
    
    Query: {query}
    
    Search Results:
    {results}
    
    Analyze the results and provide:
    1. A brief summary
    2. A list of key points
    3. A relevance score from 0 to 1
    
    {format_instructions}
    """
)

@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    """
    Query the Pinecone index using LangChain components and analyze the results.
    """
    try:
        # Perform similarity search using LangChain
        results = vectorstore.similarity_search(
            request.query,
            k=request.top_k
        )

        # Format results for the prompt
        formatted_results = "\n".join([f"- {doc.page_content}" for doc in results])

        # Generate the prompt
        _prompt = prompt.format_prompt(
            query=request.query,
            results=formatted_results,
            format_instructions=parser.get_format_instructions()
        )

        # Get the response from the LLM
        response = llm(_prompt.to_string())

        # Parse the response
        analyzed_result = parser.parse(response.content)

        # Format raw results
        raw_results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            } for doc in results
        ]

        logger.info(f"Successfully queried and analyzed results for: {request.query}")
        return QueryResponse(
            query=request.query,
            analyzed_result=analyzed_result,
            raw_results=raw_results
        )

    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint to check if the service is running.
    """
    return {"message": "Advanced LangChain Pinecone Query Microservice is operational"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
