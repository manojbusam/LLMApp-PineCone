# LLMApp-Search

# Vector Database Indexing and Querying with RAGAS Evaluation & Index Automation

## About

This project provides an end-to-end solution for indexing and querying a vector database, enhanced with RAGAS (Retrieval Augmented Generation and Summarization) evaluation and automated index management. It's designed to streamline document processing, improve information retrieval, and maintain high-quality search results.

## Key Features

1. **Vector Database Indexing:** Tools: S3, Tensorflow; Algorithms: Recursive Chunking, OpenAI ChatGPT Embedding, PineCone 
2. **Advanced Querying:** NLP-powered search mechanism for accurate results

Query:
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

4. **RAGAS Evaluation:** Continuous assessment of retrieval and summarization quality
5. **Index Automation:** Automated updates and maintenance of the index
6. **Intelligent Document Processing:** AI-driven content analysis and metadata extraction
7. **Scalability:** Designed to handle large volumes of data efficiently
8. **User-Friendly Interface:** Intuitive search and result visualization tools


 

 
