"""
Advanced LangChain-Powered Querying and Comprehensive RAGAS Evaluation Microservice

Author: Manoj Busam
Date: April 22, 2024

This FastAPI microservice enables sophisticated querying of a Pinecone vector database
using LangChain components and provides comprehensive RAGAS-based evaluation capabilities.
It uses all available RAGAS metrics for a thorough assessment of the RAG system's performance,
separated into retrieval and generation metrics.

Key features:
1. FastAPI for efficient API handling
2. LangChain integration for advanced querying and analysis
3. OpenAI embeddings and language model for processing
4. Custom prompts and output parsing for structured responses
5. Pinecone vector store for fast similarity search
6. Comprehensive RAGAS evaluation using retrieval and generation metrics
7. LangSmith integration for dashboard visualization

To use this microservice:
1. Install dependencies: pip install fastapi uvicorn langchain pinecone-client openai pydantic ragas datasets langsmith
2. Set up your Pinecone, OpenAI, and LangSmith API keys as environment variables
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
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    answer_correctness,
    aspect_critique,
    answer_similarity,
    retrieval_precision,
)
from datasets import Dataset
from typing import List, Dict
from langsmith import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Advanced LangChain Pinecone Query and Comprehensive RAGAS Evaluation Microservice", version="1.0.0")

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI Embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Initialize Pinecone vector store
vectorstore = Pinecone.from_existing_index(index_name, embeddings)

# Initialize LangSmith client
client = Client()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class AnalyzedResult(BaseModel):
    summary: str = Field(description="A brief summary of the retrieved information")
    key_points: List[str] = Field(description="List of key points extracted from the retrieved documents")
    relevance_score: float = Field(description="A score from 0 to 1 indicating the overall relevance of the results to the query")

class QueryResponse(BaseModel):
    query: str
    analyzed_result: AnalyzedResult
    raw_results: List[Dict]

class EvaluationRequest(BaseModel):
    questions: List[str]
    contexts: List[List[str]]
    answers: List[str]
    ground_truths: List[str]

class EvaluationResponse(BaseModel):
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    sample_evaluations: List[Dict]

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
        results = vectorstore.similarity_search(request.query, k=request.top_k)
        formatted_results = "\n".join([f"- {doc.page_content}" for doc in results])
        _prompt = prompt.format_prompt(
            query=request.query,
            results=formatted_results,
            format_instructions=parser.get_format_instructions()
        )
        response = llm(_prompt.to_string())
        analyzed_result = parser.parse(response.content)
        raw_results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        
        logger.info(f"Successfully queried and analyzed results for: {request.query}")
        return QueryResponse(query=request.query, analyzed_result=analyzed_result, raw_results=raw_results)

    except Exception as e:
        logger.error(f"Error querying index: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest):
    """
    Evaluate the RAG system using all available RAGAS metrics, separated into retrieval and generation metrics.
    """
    try:
        dataset = Dataset.from_dict({
            "question": request.questions,
            "contexts": request.contexts,
            "answer": request.answers,
            "ground_truth": request.ground_truths
        })
        
        # Define retrieval and generation metrics
        retrieval_metrics = [
            context_precision,
            context_recall,
            context_relevancy,
            retrieval_precision,
        ]
        
        generation_metrics = [
            faithfulness,
            answer_relevancy,
            answer_correctness,
            answer_similarity,
            aspect_critique,
        ]
        
        # Evaluate retrieval metrics
        retrieval_result = evaluate(
            dataset=dataset,
            metrics=retrieval_metrics,
            llm=llm,
            embeddings=embeddings
        )
        
        # Evaluate generation metrics
        generation_result = evaluate(
            dataset=dataset,
            metrics=generation_metrics,
            llm=llm,
            embeddings=embeddings
        )
        
        # Format results
        retrieval_dict = {metric: float(score) for metric, score in retrieval_result.items()}
        generation_dict = {metric: float(score) for metric, score in generation_result.items()}
        
        # Prepare sample evaluations for dashboard
        sample_evaluations = []
        for i in range(min(5, len(request.questions))):  # Take up to 5 samples
            sample = {
                "question": request.questions[i],
                "context": request.contexts[i],
                "answer": request.answers[i],
                "ground_truth": request.ground_truths[i],
                "retrieval_metrics": {metric: float(retrieval_result[metric][i]) for metric in retrieval_metrics},
                "generation_metrics": {metric: float(generation_result[metric][i]) for metric in generation_metrics if metric != aspect_critique},
                "aspect_critique": generation_result[aspect_critique][i]
            }
            sample_evaluations.append(sample)
        
        # Log evaluation results to LangSmith
        client.log_dataset(
            dataset_name="RAG_Evaluation",
            data=sample_evaluations
        )
        
        logger.info(f"Successfully evaluated RAG system using RAGAS metrics")
        return EvaluationResponse(
            retrieval_metrics=retrieval_dict,
            generation_metrics=generation_dict,
            sample_evaluations=sample_evaluations
        )

    except Exception as e:
        logger.error(f"Error evaluating RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint to check if the service is running.
    """
    return {"message": "Advanced LangChain Pinecone Query and Comprehensive RAGAS Evaluation Microservice is operational"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
