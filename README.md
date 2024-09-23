#  Search Engine with Vector Database Management (Index, Query, Evaluate, Automate & Deploy in K8)

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

Query:
```
sample_evaluation_request = EvaluationRequest(
    questions=[
        "What are the main advantages of vector databases?",
        "How do vector databases improve search capabilities?",
        "What are some applications of vector databases in AI?",
    ],
    contexts=[
        [
            "Vector databases excel at similarity search in high-dimensional spaces, making them ideal for AI and machine learning applications. They can efficiently store and query large-scale embedding data, enabling fast and accurate retrieval of similar items.",
            "One of the main advantages of vector databases is their ability to understand the semantic meaning of data. This allows for more nuanced and context-aware querying compared to traditional keyword-based search methods."
        ],
        [
            "Vector databases improve search capabilities by using mathematical representations (vectors) of data points. This allows for similarity searches based on the actual content and meaning of the data, rather than just keyword matching.",
            "By storing data as vectors, these databases can quickly find similar items in large datasets, which is particularly useful for recommendation systems, image search, and natural language processing tasks."
        ],
        [
            "Vector databases are widely used in AI applications such as recommendation systems, where they can quickly find similar items based on user preferences or item features.",
            "In natural language processing, vector databases store word or sentence embeddings, enabling efficient semantic search and language understanding tasks.",
            "Computer vision applications use vector databases to store and query image embeddings, facilitating tasks like image similarity search and object recognition."
        ]
    ],
    answers=[
        "The main advantages of vector databases include efficient similarity search in high-dimensional spaces, fast querying of large-scale datasets, and improved semantic understanding of data.",
        "Vector databases improve search capabilities by enabling similarity searches based on content meaning and quickly finding similar items in large datasets, which is useful for various AI applications.",
        "Vector databases are used in AI applications such as recommendation systems, natural language processing for semantic search, and computer vision for image similarity and object recognition tasks."
    ],
    ground_truths=[
        "Vector databases offer advantages such as efficient similarity search in high-dimensional spaces, fast querying of large-scale datasets, improved semantic understanding, and suitability for AI and machine learning applications.",
        "Vector databases enhance search capabilities through semantic understanding, efficient similarity computations, support for high-dimensional data, and the ability to quickly find similar items in large datasets, improving various AI applications.",
        "Vector databases are applied in AI for recommendation systems, natural language processing (including semantic search and language understanding), and computer vision tasks like image similarity search and object recognition."
    ]
)
```

Response:
```
{
  "retrieval_metrics": {
    "context_precision": 0.92,
    "context_recall": 0.89,
    "context_relevancy": 0.95,
    "retrieval_precision": 0.91
  },
  "generation_metrics": {
    "faithfulness": 0.88,
    "answer_relevancy": 0.93,
    "answer_correctness": 0.90,
    "answer_similarity": 0.87
  },
  "sample_evaluations": [
    {
      "question": "What are the main advantages of vector databases?",
      "context": [
        "Vector databases excel at similarity search in high-dimensional spaces, making them ideal for AI and machine learning applications. They can efficiently store and query large-scale embedding data, enabling fast and accurate retrieval of similar items.",
        "One of the main advantages of vector databases is their ability to understand the semantic meaning of data. This allows for more nuanced and context-aware querying compared to traditional keyword-based search methods."
      ],
      "answer": "The main advantages of vector databases include efficient similarity search in high-dimensional spaces, fast querying of large-scale datasets, and improved semantic understanding of data.",
      "ground_truth": "Vector databases offer advantages such as efficient similarity search in high-dimensional spaces, fast querying of large-scale datasets, improved semantic understanding, and suitability for AI and machine learning applications.",
      "retrieval_metrics": {
        "context_precision": 0.95,
        "context_recall": 0.92,
        "context_relevancy": 0.97,
        "retrieval_precision": 0.94
      },
      "generation_metrics": {
        "faithfulness": 0.91,
        "answer_relevancy": 0.95,
        "answer_correctness": 0.93,
        "answer_similarity": 0.89
      },
      "aspect_critique": {
        "overall_score": 0.92,
        "reasoning": 0.90,
        "relevance": 0.94,
        "coherence": 0.93,
        "conciseness": 0.91
      }
    },
    {
      "question": "How do vector databases improve search capabilities?",
      "context": [
        "Vector databases improve search capabilities by using mathematical representations (vectors) of data points. This allows for similarity searches based on the actual content and meaning of the data, rather than just keyword matching.",
        "By storing data as vectors, these databases can quickly find similar items in large datasets, which is particularly useful for recommendation systems, image search, and natural language processing tasks."
      ],
      "answer": "Vector databases improve search capabilities by enabling similarity searches based on content meaning and quickly finding similar items in large datasets, which is useful for various AI applications.",
      "ground_truth": "Vector databases enhance search capabilities through semantic understanding, efficient similarity computations, support for high-dimensional data, and the ability to quickly find similar items in large datasets, improving various AI applications.",
      "retrieval_metrics": {
        "context_precision": 0.93,
        "context_recall": 0.90,
        "context_relevancy": 0.96,
        "retrieval_precision": 0.92
      },
      "generation_metrics": {
        "faithfulness": 0.89,
        "answer_relevancy": 0.94,
        "answer_correctness": 0.91,
        "answer_similarity": 0.88
      },
      "aspect_critique": {
        "overall_score": 0.91,
        "reasoning": 0.89,
        "relevance": 0.93,
        "coherence": 0.92,
        "conciseness": 0.90
      }
    },
    {
      "question": "What are some applications of vector databases in AI?",
      "context": [
        "Vector databases are widely used in AI applications such as recommendation systems, where they can quickly find similar items based on user preferences or item features.",
        "In natural language processing, vector databases store word or sentence embeddings, enabling efficient semantic search and language understanding tasks.",
        "Computer vision applications use vector databases to store and query image embeddings, facilitating tasks like image similarity search and object recognition."
      ],
      "answer": "Vector databases are used in AI applications such as recommendation systems, natural language processing for semantic search, and computer vision for image similarity and object recognition tasks.",
      "ground_truth": "Vector databases are applied in AI for recommendation systems, natural language processing (including semantic search and language understanding), and computer vision tasks like image similarity search and object recognition.",
      "retrieval_metrics": {
        "context_precision": 0.96,
        "context_recall": 0.93,
        "context_relevancy": 0.98,
        "retrieval_precision": 0.95
      },
      "generation_metrics": {
        "faithfulness": 0.92,
        "answer_relevancy": 0.96,
        "answer_correctness": 0.94,
        "answer_similarity": 0.90
      },
      "aspect_critique": {
        "overall_score": 0.93,
        "reasoning": 0.91,
        "relevance": 0.95,
        "coherence": 0.94,
        "conciseness": 0.92
      }
    }
  ]
}
```
   
5. **AIRFLOW Index Automation:** Automated updates and maintenance of the index
This Airflow DAG automates a vector database pipeline with four main tasks: listing S3 files, indexing documents, performing queries, and evaluating results using RAGAS metrics. The tasks are executed sequentially, with dependencies set as: list_files_task >> indexing_task >> query_task >> evaluate_task

7. **Docker Containerization & Registry** AI-driven content analysis and metadata extraction
8. **Kubernetes Deployment:** Designed to handle large volumes of data efficiently
 


 

 
