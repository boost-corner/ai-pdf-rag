# 📄 AI That Reads and Understands PDFs [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boost-corner/ai-pdf-rag/blob/main/pdf_rag.ipynb)

This tutorial demonstrates how to build an AI-powered workflow for reading and understanding PDF documents using LangChain, LangGraph, OpenAI, and PyPDF.
You’ll learn how to extract content, generate semantic embeddings, search document chunks by meaning, and generate context-aware answers — all inside a modular Retrieval-Augmented Generation (RAG) pipeline.

---

## ✅ What It Does

This project builds an AI-powered assistant that can:

- Load and parse PDF documents
- Split content into semantic chunks
- Generate embeddings using OpenAI
- Perform semantic search over document content
- Answer user questions based solely on the PDF's context
- Run the entire Retrieval-Augmented Generation (RAG) workflow with LangGraph

---

## ⚙️ Technologies Used

- **LangGraph** – for building modular, stateful RAG workflows  
- **LangChain** – for managing document loading, chunking, and vector stores  
- **OpenAI API** – `text-embedding-3-large` for embeddings, `gpt-4` for answering  
- **PyPDF** – for extracting content from PDF files  
- **Jupyter Notebook** – for running the interactive demo  

---

## 📦 Requirements

- OpenAI API Key
- LangSmith account (optional but recommended for tracing/debugging)

---

## 💼 Use Cases

- 🔍 Semantic document search  
- 📚 Research paper analysis  
- ⚖️ Legal document review  
- 💼 Company knowledge base exploration  
- 🤖 Building custom intelligent PDF chatbots  

---

## 👥 This Project is Perfect For...

- **Developers** who want to integrate document Q&A into apps  
- **Researchers** who need fast insights from academic PDFs  
- **Legal/Compliance Teams** analyzing large contracts or policies  
- **Product Teams** building LLM tools with real-world documents  
- **Educators** creating intelligent reading assistants  
