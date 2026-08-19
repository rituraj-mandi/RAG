# 🤖 Voice-Enabled RAG

A **Voice-Enabled Retrieval-Augmented Generation (RAG)** system that allows users to ask questions using voice and receive answers based on a provided knowledge dataset.

## ✨ Features

* 🎙️ Voice-based question input
* 🔊 AI-generated responses
* 🔎 Semantic document retrieval
* 🧠 Retrieval-Augmented Generation
* 🌐 Web-based interface
* ⚡ Fast and context-aware answers

## 🏗️ Pipeline

```text
Voice Input
    ↓
Speech-to-Text
    ↓
Query Processing
    ↓
Document Retrieval
    ↓
Context + LLM
    ↓
Generated Answer
    ↓
Text-to-Speech
```

## 🛠️ Tech Stack

* **Frontend:** HTML / CSS / JavaScript
* **Backend:** Python / FastAPI
* **RAG:** LangChain
* **Vector Database:** PostgreSQL + pgvector
* **LLM:** Gemini
* **Dataset:** MSMARCO-XI
* **Deployment:** Vercel + AWS EC2

## 🚀 Setup

```bash
git clone https://github.com/rituraj-mandi/V-RAG.git
cd V-RAG
```

Install dependencies and configure your `.env` file with the required API keys.

## 📌 Purpose

Built as a prototype for a **Voice-Enabled RAG system**, combining speech processing, information retrieval, and generative AI into a single application.
