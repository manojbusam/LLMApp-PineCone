#  Search Engine with Vector Database Management with Evaluation & Deployment in K8

## About

This project provides an end-to-end solution for indexing and querying a vector database, enhanced with RAGAS (Retrieval Augmented Generation and Summarization) evaluation and automated index management. It's designed to streamline document processing, improve information retrieval, and maintain high-quality search results.

## Key Features
1. **Vector Database Indexing:** Tools: S3, Tensorflow , PineCone ; Algorithms: Recursive Chunking, OpenAI ChatGPT Embedding 
2. **Advanced Querying:** Libs: Fast API, Prompt, Parser ; Tools: Pinecone ; NLP-powered search mechanism for accurate results
3. **RAGAS Evaluation:** Continuous assessment of Retrieval and Generation quality
4. **AIRFLOW Index Automation:** Automated updates and maintenance of the index
5. **Docker Containerization & Registry** AI-driven content analysis and metadata extraction
6. **Kubernetes Deployment:** Designed to handle large volumes of data efficiently

## Key Feature Explaination

1. **Vector Database Indexing:** S3 stores documents, TensorFlow processes data, and Pinecone indexes vectors. Recursive chunking splits texts, while OpenAI ChatGPT generates embeddings for semantic search in Pinecone.
2. **Advanced Querying:** FastAPI handles API requests, Prompt templates queries, Parser structures outputs; Pinecone enables vector search; NLP-powered mechanism uses embeddings for accurate semantic retrieval.

Sample Query:
```
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key benefits of using vector databases for information retrieval?", "top_k": 3}'
```

Result:
```
{
  "query": "What are the key benefits of using vector databases for information retrieval?",
  "analyzed_result": {
    "summary": "Vector databases offer significant advantages for information retrieval, including improved semantic search capabilities, faster query processing, and the ability to handle high-dimensional data efficiently.",
    "key_points": [
      "Enhanced semantic search and similarity matching",
      "Faster query processing compared to traditional databases",
      "Efficient handling of high-dimensional data",
      "Scalability for large datasets",
      "Ability to perform complex queries based on content similarity"
    ],
    "relevance_score": 0.95
  },
  "raw_results": [
    {
      "content": "Vector databases revolutionize information retrieval by enabling semantic search capabilities. Unlike traditional databases, vector databases can understand the context and meaning behind queries, leading to more accurate and relevant results.",
      "metadata": {
        "source": "article_1.txt",
        "author": "Jane Doe"
      }
    },
    {
      "content": "One of the primary advantages of vector databases is their ability to process queries much faster than traditional relational databases, especially for complex similarity searches in high-dimensional spaces.",
      "metadata": {
        "source": "tech_blog.md",
        "date": "2024-03-15"
      }
    },
    {
      "content": "Vector databases excel at handling high-dimensional data, making them ideal for applications in machine learning, natural language processing, and image recognition. They can efficiently store and query embeddings, which are crucial for these advanced AI applications.",
      "metadata": {
        "source": "research_paper.pdf",
        "institution": "Tech University"
      }
    }
  ]
```

3. **RAGAS Evaluation:** Continuous assessment of retrieval and generation quality using:

   1. Retrieval Metrics:
      - context_precision: Measures how relevant the retrieved documents are to the query.
      - context_recall: Assesses how well the retrieved documents cover the information needed to answer the query.
      - context_relevancy: Evaluates the overall relevance of the retrieved context to the query.
      - retrieval_precision: Measures the precision of the retrieval step in the RAG pipeline.

   2. Generation Metrics:
      - faithfulness: Assesses how well the generated answer aligns with the provided context.
      - answer_relevancy: Evaluates how relevant the generated answer is to the given question.
      - answer_correctness: Measures the factual correctness of the generated answer.
      - answer_similarity: Compares the similarity between the generated answer and the ground truth.
      - aspect_critique: Provides a detailed critique of various aspects of the RAG system's performance in generating answers, including:
        * overall_score: An aggregate score of all aspects.
        * reasoning: Evaluates the logical flow and argumentation in the answer.
        * relevance: Assesses how well the answer addresses the specific question asked.
        * coherence: Measures the clarity and logical consistency of the answer.
        * conciseness: Evaluates whether the answer is appropriately brief and to the point.

   These metrics provide a comprehensive evaluation of both the retrieval and generation components of the RAG system. By continuously monitoring these metrics, you can identify areas for improvement, track the system's performance over time, and ensure high-quality responses to user queries.
   
4. **AIRFLOW Index Automation:** Automated updates and maintenance of the index
5. **Docker Containerization & Registry** AI-driven content analysis and metadata extraction
6. **Kubernetes Deployment:** Designed to handle large volumes of data efficiently
 


 

 
