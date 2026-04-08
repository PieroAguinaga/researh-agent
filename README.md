<div align="center">
  <br/>
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangGraph-multi--agent-1C3C3C?style=flat-square"/>
  <img src="https://img.shields.io/badge/Azure_OpenAI-GPT--4o-0078D4?style=flat-square&logo=microsoftazure&logoColor=white"/>
  <img src="https://img.shields.io/badge/Supabase-pgvector-3FCF8E?style=flat-square&logo=supabase&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square"/>
  <br/><br/>

  <h1>⬡ IATA</h1>
  <h3>Intelligent Agent for Technology Acquisition</h3>
  <p>
    A <strong>production-grade multi-agent AI system</strong> that autonomously retrieves,<br/>
    summarizes, and synthesizes the latest scientific research papers.
  </p>

  <br/>

  > Built with **LangGraph** · **Azure OpenAI** · **Supabase pgvector** · **arXiv API** · **Semantic Scholar**

  <br/>
</div>

---

## What is IATA?

Choosing the right technology stack requires hours of manual research across papers, docs, and blogs. **IATA eliminates that overhead.**

Given a natural language query, the agent autonomously:

1. **Routes** the request to the right specialist agent via a supervisor node
2. **Searches** arXiv and Semantic Scholar concurrently for relevant papers
3. **Summarizes** findings into structured, actionable insights using an LLM
4. **Remembers** the full conversation across sessions via Supabase persistent memory

The result is a research-backed assistant that feels like talking to a well-read colleague.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Chat Interface                    │
│              (Flask + Vanilla JS/CSS)               │
└───────────────────────┬─────────────────────────────┘
                        │ POST /api/chat
┌───────────────────────▼─────────────────────────────┐
│                   Flask REST API                     │
│          Loads Supabase history → invokes graph      │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              LangGraph Multi-Agent Graph             │
│                                                     │
│   START → [supervisor_node] ──────────────► END     │
│                    │                                │
│          ┌─────────┴──────────┐                    │
│          ▼                    ▼                     │
│   [search_agent]     [summarizer_agent]             │
│   search_papers()    summarize_paper()              │
│   arXiv + SS         LLM + JsonOutputParser         │
│          │                    │                     │
│          └─────────┬──────────┘                    │
│                    ▼                                │
│             [supervisor_node]  ◄── re-evaluates     │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼
        ▼               ▼               
  Supabase DB      arXiv API    
  (3 tables)       papers       
  - history
  - papers
  - embeddings
```

### Multi-Agent Design

| Node | Role | Tools |
|---|---|---|
| `supervisor_node` | Reads the user message and routes to the right specialist, or ends the turn | LLM reasoning only |
| `search_agent` | Finds papers from arXiv and Semantic Scholar | `search_papers` |
| `summarizer_agent` | Produces structured summaries for each retrieved paper | `summarize_paper`, `summarize_multiple_papers` |

After each specialist completes, control returns to the supervisor, which can chain specialists or finish.

---

## Features

- 🤖 **Multi-agent orchestration** with LangGraph — supervisor routes to specialized nodes
- 🔍 **Dual-source retrieval** — arXiv and Semantic Scholar queried in parallel
- 📝 **Structured summarization** — LLM + Pydantic schema produces consistent JSON output
- 🧠 **Persistent memory** — full conversation history saved to Supabase across sessions
- 🗄️ **Vector store ready** — `paper_embeddings` table with pgvector for future RAG
- ⚙️ **Tool call transparency** — the UI shows which agents and tools ran per turn
- 🔒 **Type-safe config** — Pydantic Settings with `.env` file validation
- 🧪 **Unit tests** — core logic tested without API calls using mocks

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Azure OpenAI (GPT-4o via `AzureChatOpenAI`) |
| Agent orchestration | LangGraph 0.2 — multi-agent `StateGraph` |
| LLM framework | LangChain 0.2 |
| Database | Supabase (PostgreSQL + pgvector) |
| Paper APIs | arXiv Python client, Semantic Scholar Graph API |
| Backend | Flask 3 |
| Config | Pydantic Settings |
| Testing | pytest |

---


## Quickstart — Windows

### 1. Clone and create environment

```bat
git clone https://github.com/PieroAguinaga/researh-agent.git
cd iata
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure environment variables

```bat
copy .env.example .env
```

Open `.env` and fill in your values:

```env
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key-here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
```

### 3. Set up Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor → New Query**
3. Paste the entire contents of `supabase/schema.sql` and click **Run**

This creates three tables (`conversation_history`, `papers`, `paper_embeddings`), the pgvector index, and the `match_paper_embeddings` RPC function.

### 4. Verify all connections

```bat
python tests/setup_verification.py
```

Expected output:
```
── IATA Setup Verification ──────────────────────────────

  ✓  Environment variables
  ✓  Azure OpenAI connection
  ✓  Supabase connection
  ✓  Supabase tables exist
  ✓  arXiv API reachable

── Result: 5/5 checks passed ─────────────────────

  Everything looks good. Run: python run.py
```

### 5. Run

```bat
python -m api.app
```

Open [http://localhost:5000](http://localhost:5000)

---

## Usage Examples

**Find and summarize papers:**
```
Find the latest papers on retrieval-augmented generation
```

**Get a summary:**
```
Summarize the top papers on mixture of experts models
```

**Explore trends:**
```
What are the most recent advances in efficient transformers?
```

**Cross-session memory:**
Every conversation is persisted in Supabase. Returning to the same `thread_id` resumes exactly where you left off.

---

## API Reference

### `POST /api/chat`

```json
// Request
{ "message": "Find papers on sparse attention", "thread_id": "optional-uuid" }

// Response
{
  "reply": "Here are the papers I found...",
  "thread_id": "abc-123",
  "tool_calls_used": ["search_papers", "summarize_multiple_papers"]
}
```

### `GET /api/papers/search`

```
GET /api/papers/search?q=mixture+of+experts&max_results=8
```

### `DELETE /api/chat/<thread_id>`

Clears conversation history for the given thread from Supabase.

---


## Roadmap

| Feature | Status |
|---|---|
| arXiv + Semantic Scholar retrieval | ✅ Done |
| LLM structured summarization | ✅ Done |
| Supabase persistent memory | ✅ Done |
| Multi-agent LangGraph graph | ✅ Done |
| Flask API + Chat UI | ✅ Done |
| RAG Q&A over indexed papers | 🔄 Planned |
| Trend detection agent node | 🔄 Planned |
| Streaming responses (SSE) | 🔄 Planned |
| LangGraph Studio visualization | 🔄 Planned |
| Evaluation harness (RAGAS) | 🔄 Planned |

---

## Skills Demonstrated

This project was designed to showcase competencies relevant to **AI/ML Engineering** and **Backend Engineering** roles:

| Skill | Implementation |
|---|---|
| LLM orchestration | LangGraph `StateGraph` with supervisor routing |
| Multi-agent design | Specialized nodes with conditional edges |
| Tool use / function calling | LangChain `@tool` + `bind_tools` |
| Prompt engineering | Structured JSON output with Pydantic schemas |
| Vector database | Supabase pgvector with IVFFlat index |
| Persistent memory | Supabase-backed conversation history |
| External API integration | arXiv client |
| Clean architecture | Separation of agent, API, config, and frontend layers |
| Type safety | Pydantic Settings + dataclasses throughout |
| Error handling | Graceful degradation on API failures + retries |
| Testing | Unit tests with mocking (no API keys required) |

---

## License

[MIT](LICENSE) © 2025
