# Mini-RAG: Document Chatbot 🤖📄

A minimalist, high-performance Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF document and have context-grounded, real-time conversations with its contents. Built with **Streamlit**, **LangChain**, and **ChromaDB**.

Live Demo: *[Insert your live share.streamlit.io link here]*

---

## 🚀 Key Features

* **Session-State Cost Optimization:** The application caches the vector database and processed document embedding splits within `st.session_state`. This ensures document embedding processes happen exactly once per file upload, avoiding massive token overhead and redundant API costs on follow-up chat messages.
* **Isolated Multi-User Architecture:** Utilizes an ephemeral, in-memory Chroma instance built dynamically per session. If multiple users access the app concurrently via the web, their document contexts remain completely isolated and secure.
* **Context-Bounded Guardrails:** Engineered with explicit system prompts instructing the LLM to strictly ground answers within the retrieved PDF context. The model will actively refuse to hallucinate or pull outside training data when probed with out-of-context or adversarial questions.
* **Streamlit Native Chat UI:** Implements a smooth, familiar chat interface utilizing `st.chat_input` and `st.chat_message` asynchronous state workflows.

---

## 🛠️ Tech Stack

* **Frontend UI:** Streamlit
* **LLM Orchestration Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings Model:** OpenAI `text-embedding-3-small`
* **Frontier Model:** OpenAI `gpt-4o-mini`
* **Document Parsing:** PyPDF

---

## 📦 Architecture Workflow

1. **Document Ingestion:** User drops a PDF into the Streamlit interface.
2. **Text Chunking:** The document is loaded via `PyPDFLoader` and broken down using `RecursiveCharacterTextSplitter` (configured with an optimized chunk size and boundary overlap).
3. **Vectorization:** Text chunks are transformed into 1536-dimension dense vectors via OpenAI's embeddings API.
4. **Vector Indexing:** The vectors are stored in a local, ephemeral session-bound Chroma index.
5. **Similarity Search:** When a user queries the chat, a top-k semantic lookup extracts the most relevant text passages.
6. **Context Enrichment:** The prompt template pipes the raw query alongside the pulled text segments to the LLM to formulate a deterministic, factual response.

---

## ⚙️ Getting Started & Installation

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/atharva-rajguru/Mini-RAG.git](https://github.com/atharva-rajguru/Mini-RAG.git)
cd Mini-RAG
