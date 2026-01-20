<img width="1512" height="816" alt="StirFry Contemplating" src="https://github.com/user-attachments/assets/4b1df3e5-ac4d-4b47-aabb-b8550410784c" />
<img width="2268" height="1124" alt="image" src="https://github.com/user-attachments/assets/3b99d14b-b758-4945-91ed-246598a4e895" />

# StirFry 🍳  
Marcello’s stateful financial researcher — a Streamlit app that combines a portfolio-aware chat assistant with a custom-built LangGraph-style multi-agent workflow.

## What this repo is
StirFry is a financial assistant that can operate in two modes:

- **Conversational (“Geeked”)**: quick answers, tool-augmented chat, and lightweight reasoning.
- **Locked [Research] (“deep_research”)**: a structured multi-step research loop that plans, executes tool calls, reflects on gaps, and synthesizes a final report.

The app is intentionally **stateful**:
- It stores and reads **user profile context** and a **live portfolio** from Supabase.
- It maintains chat history in Streamlit session state.
- It uses a LangGraph checkpoint backend (Postgres) to support persistent graph execution.

The UI is also custom-styled (avatars, chat styling, a “System Logs” status panel, and step banners) to make the agent’s internal workflow visible while keeping the final output readable.

---

## Intent (why I built this)
This project is primarily a learning build!

The goal is to gain **firsthand experience** designing and implementing **multi-agent systems from scratch**—not just calling frameworks, but making real architectural decisions and dealing with the consequences.

Key skills and themes I’m practicing here:
- Designing an agent system around **constraints** (token limits, tool reliability, API error modes, recursion depth, latency).
- Making **tradeoffs** between “clean architecture” vs. “shipping something usable.”
- Defining **scope boundaries** so the system stays controllable (e.g., a capped research refinement loop instead of infinite self-improvement).
- Building a custom research workflow that behaves like a small team:
  - a planner that decomposes the objective,
  - an executor that gathers evidence,
  - a reflector/critic that checks for gaps,
  - a synthesizer that produces a final answer.

---

## High-level architecture
There are three major layers:

### 1) Streamlit application layer (`app.py`)
This is the user-facing product surface.

It handles:
- App layout, theming, and the chat UI.
- A “Quick Update” form to manually inform StirFry whether I made changes to my portfolio (buy/sell).
- Portfolio state reads/writes via Supabase.
- Routing user prompts into the agent system.
- Streaming model output into the chat window while also showing internal status logs.

The chat loop is built around Streamlit session state:
- `st.session_state.messages` stores the conversation.
- Messages are rendered with a sanitizer that strips internal tags like `[THINKING]...[/THINKING]`.

### 2) Tooling + data layer (`tools.py` + Supabase + external APIs)
Tools provide the system’s “hands.”

The key tools include:
- **Market + fundamentals** via `yfinance` (`get_market_data`, `get_company_health`).
- **Macro indicators** via the FRED API (`get_macro_data`).
- **Portfolio valuation** from Supabase holdings + live prices (`get_my_portfolio`).
- **Profile memory editing** stored in Supabase (`update_profile_notes`).
- **Web search** via Tavily (`TavilySearchResults`).
- A strict **calculator tool** (`calculate_math`) to prevent hallucinated arithmetic.

Supabase is used for:
- `profiles` table: stores a long-lived `base_context` string per user.
- `portfolio` table: stores holdings (`ticker`, `shares`, `avg_cost`) keyed by `user_id`.

### 3) The custom LangGraph architecture (`graph.py` + `research_engine.py`)
This repo isn’t “LangGraph as a black box.” It’s my attempt to build a graph-based multi-agent system with clear state, routing, and guardrails.

At the center is a **StateGraph** with a typed state (`AgentState`) containing:
- messages
- user_id
- mode (conversational vs deep_research)
- research_session_id
- research_plan
- loop_count
- user_goal

#### Core nodes
**Advisor node**
- Default mode.
- Responds like a normal assistant but can call tools.
- Pulls user context + portfolio holdings into the system prompt.

**Research Planner node**
- Triggered when prompts are routed into deep research mode (e.g., prefixed by `DEEP_RESEARCH:`).
- Creates a structured plan with subtasks and tool selection.
- Starts a new research session id and stores the plan in state.

**Research Executor node**
- Runs planned subtasks using a tool-enabled model.
- Designed to prioritize recency and avoid stale data.

**Research Reflector node**
- Acts like a critic/editor.
- Checks whether the gathered evidence is sufficient.
- If incomplete, it proposes refined tasks and loops back to execution.
- Has explicit safeguards:
  - loop cap (`MAX_LOOPS`)
  - “no-progress” detection (don’t keep regenerating the same plan)
  - robust JSON extraction and a “fail closed” behavior to avoid infinite loops

**Research Synthesis node**
- Produces the final “report-style” answer using the collected context + cleaned history.
- Resets mode back to conversational after completion.

#### Why build the graph this way?
Because multi-agent systems fail in predictable ways unless you design for them:
- Tool calls can break message format requirements (OpenAI 400s).
- Reflection loops can spiral or re-run the same plan forever.
- Long chat histories can blow context windows and crash the session.
- “Smart” agents still need boring engineering: validation, caps, routing logic, and deterministic fallback behavior.

This repo treats those failure modes as part of the product design, not edge cases.

---

## Memory & history hygiene (one of the main engineering constraints)
This project includes a “robust memory sanitizer” (`get_clean_history`) that trims and validates message history before sending it to the model.

The point is practical: when an LLM emits tool calls, the model/tool message sequence must stay consistent, or you can trigger API errors. The sanitizer:
- keeps only a bounded number of recent messages,
- ensures tool call ↔ tool result pairs are valid,
- drops orphan tool messages and malformed tool-call sequences.

This is one of the places where the repo is explicitly about **systems constraints** rather than “prompting harder.”
