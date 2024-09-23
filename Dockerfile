# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into the container
COPY . .

# Set environment variables
ENV PINECONE_API_KEY=your_pinecone_api_key
ENV PINECONE_ENV=your_pinecone_environment
ENV PINECONE_INDEX_NAME=your_pinecone_index_name
ENV OPENAI_API_KEY=your_openai_api_key
ENV LANGCHAIN_API_KEY=your_langchain_api_key

# Expose the port that the FastAPI app will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
