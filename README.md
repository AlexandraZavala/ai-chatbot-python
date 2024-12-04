# 🤖 **Code Chatbot with RAG (Retrieval and Generation)**

This project implements a chatbot that allows interaction with a codebase using retrieval and generation techniques. Pinecone is used for document indexing and OpenAI's GPT-3/4 is used to answer questions about the code.

## 🚀 **Technologies Used**

- **FastAPI** 🖥️: To create the web server and handle API requests.
- **LangChain** 🔗: To integrate tools like Pinecone and embeddings with AI models.
- **Pinecone** 🌲: For vector indexing of code files.
- **Sentence-Transformers** 🧠: To generate embeddings of code snippets.
- **OpenAI GPT** 💬: For generating answers to code-related queries.
- **HuggingFace** 🤗: For text embeddings and natural language processing.
- **Google Generative AI** 🏗️: For generating enriched content using images and text.
- **Pydantic** 📦: For input data validation in the API.
- **Git** 🔧: To clone code repositories and process them.
- **PIL (Python Imaging Library)** 🖼️: To process images as part of the queries.
- **dotenv** 🌍: To load environment variables.
- **Logging** 📝: To manage application logs.

## 🛠️ **What Was Done?**

1. **Repository Cloning**: You can clone GitHub repositories and extract the content of code files. This allows dynamic analysis of codebases.
2. **Pinecone Indexing**: Code snippets are converted into embeddings and stored in Pinecone, enabling efficient retrieval of relevant snippets.
3. **Query Responses (RAG)**: Using embeddings, the system retrieves relevant code snippets and generates answers to user queries based on the code.
4. **AI-Generated Responses**: Integration with OpenAI allows for detailed and precise answers about the code, with the option to include images as additional context.
