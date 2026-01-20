import os
import operator
import uuid
import json
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, List, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from supabase import create_client

# IMPORTS
from tools import tools, TOOL_NAME_MAP
from research_engine import DeepResearchEngine

load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
research_engine = DeepResearchEngine()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    mode: Literal["conversational", "deep_research"]
    research_session_id: str | None
    research_plan: dict | None
    loop_count: int
    user_goal: str


# --- HELPER: ROBUST MEMORY SANITIZER ---
def get_clean_history(messages: Sequence[BaseMessage], max_messages: int = 10) -> List[BaseMessage]:
    """Sanitizes history to prevent OpenAI 400 errors."""
    recent = messages[-max_messages:]
    sanitized = []
    skip_indices = set()
    for i, msg in enumerate(recent):
        if i in skip_indices:
            continue
        # guard attribute access
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            num_calls = len(msg.tool_calls)
            if i + num_calls >= len(recent):
                continue
            is_valid = all(isinstance(recent[i + j], ToolMessage) for j in range(1, num_calls + 1))
            if is_valid:
                sanitized.append(msg)
                for j in range(1, num_calls + 1):
                    sanitized.append(recent[i + j])
                    skip_indices.add(i + j)
        elif isinstance(msg, ToolMessage):
            continue
        else:
            sanitized.append(msg)
    return sanitized


def get_context(user_id: str):
    """Fetches User Bio + Live Portfolio"""
    try:
        prof = supabase.table("profiles").select("base_context").eq("id", user_id).maybe_single().execute()
        bio = prof.data.get("base_context", "") if prof.data else ""
        holdings = supabase.table("portfolio").select("ticker,shares,avg_cost").eq("user_id", user_id).execute().data or []
        holdings_text = "\n".join(f"{h['ticker']}: {float(h['shares']):.4f} sh @ ${float(h['avg_cost']):.2f}" for h in holdings)
    except Exception:
        bio, holdings_text = "", "No positions."
    return f"USER BIO:\n{bio}\n\nPORTFOLIO HOLDINGS:\n{holdings_text}"


# --- NODES ---


def advisor_node(state: AgentState):
    """CASUAL MODE: Simple Chat + Tools"""
    context = get_context(state["user_id"])

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

    # SYSTEM PROMPT UPDATE: Enforce Math Tool
    sys_msg = SystemMessage(content=f"""
    You are StirFry. Marcello's stateful financial assistant.
    Current Date: {datetime.now().strftime('%B %d, %Y')}

    USER CONTEXT:
    {context}

    CRITICAL RULES:
    1. **NO MENTAL MATH.** If you need to add, subtract, or calculate Net Worth, you MUST use the `calculate_math` tool.
    2. Speak the TRUTH. If data is missing, say it. Do not guess numbers.
    3. Style: Concise, helpful, "finance bro" but professional.
    """)

    return {"messages": [model.invoke([sys_msg] + get_clean_history(state["messages"]))]}


def research_planner_node(state: AgentState):
    """RESEARCH MODE: Step 1 - Plan"""
    user_id = state["user_id"]
    raw_input = state["messages"][-1].content
    objective = raw_input.replace("DEEP_RESEARCH:", "").strip()
    sid = f"{user_id}_{uuid.uuid4().hex[:8]}"

    date_context = f"Current Date: {datetime.now().strftime('%B %d, %Y')}. Focus on LATEST 2025/2026 data."
    enhanced_objective = f"{objective} ({date_context})"

    res = research_engine.execute_research(sid, enhanced_objective, None, [t.name for t in tools])

    return {
        "messages": [AIMessage(content=f"[THINKING]Plan Created: {len(res['plan']['subtasks'])} steps.[/THINKING]")],
        "mode": "deep_research",
        "research_session_id": sid,
        "research_plan": res["plan"],
        "loop_count": 0,
        "user_goal": objective,
    }


def research_executor_node(state: AgentState):
    """RESEARCH MODE: Step 2 - Execute"""
    plan = state["research_plan"]
    if not plan:
        return {"messages": [AIMessage(content="Error: Research plan missing. Switching to casual mode.")], "mode": "conversational"}

    model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)
    today = datetime.now().strftime("%B %d, %Y")
    sys = SystemMessage(content=f"""
    Current Date: {today}.
    Execute tasks: {plan['subtasks']}
    CRITICAL: You are in {today.split(',')[1]}. Data from 2023 is OLD. Always search for 'TTM', '2025', or '2026' data.
    """)
    return {"messages": [model.invoke([sys] + get_clean_history(state["messages"]))]}


def research_reflector_node(state: AgentState):
    """RESEARCH MODE: Step 3 - Reflect (robust, non-looping)"""
    plan = state.get("research_plan")
    if not plan:
        return {"next_step": "synthesize"}

    # Safety params
    MAX_LOOPS = 3
    loop_count = int(state.get("loop_count", 0))

    # Collect recent ToolMessage evidence (slightly wider window)
    recent_msgs = state["messages"][-8:]
    evidence_text = ""
    for m in recent_msgs:
        if isinstance(m, ToolMessage):
            # m.name might not exist in all ToolMessage implementations; guard it
            name = getattr(m, "name", getattr(m, "tool_name", "tool"))
            evidence_text += f"Tool ({name}): {getattr(m, 'content', '')}\n"

    # If no evidence yet, continue executing
    if not evidence_text:
        return {"next_step": "execute", "loop_count": loop_count}

    # Ask the critic to produce STRICT JSON only
    critic_model = ChatOpenAI(model="gpt-4o", temperature=0)
    today = datetime.now().strftime("%B %d, %Y")
    prompt = f"""
Current Date: {today}.
OBJECTIVE: {plan.get('objective', 'N/A')}
EVIDENCE GATHERED: {evidence_text}

Return ONLY valid JSON with the keys:
{{"status": "complete" or "incomplete",
 "reasoning": "<short plain text>",
 "new_tasks": ["task1", "task2"] }}

Do not include any explanation outside the JSON.
"""

    # invoke and try safe parse
    try:
        res = critic_model.invoke([SystemMessage(content=prompt)])
        raw = (res.content or "").strip()
        # Robust extraction: find first "{" ... "}" block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in critic response")
        payload_text = raw[start:end]
        payload = json.loads(payload_text)
    except Exception as e:
        # If parsing failed, treat as complete to avoid endless loops.
        return {
            "messages": [AIMessage(content=f"[THINKING]Reflection failed to parse JSON ({e}). Marking complete.[/THINKING]")],
            "next_step": "synthesize",
            "loop_count": loop_count,
        }

    status = payload.get("status", "complete")
    new_tasks = payload.get("new_tasks", [])
    reasoning = payload.get("reasoning", "")

    # If status is incomplete, check for progress vs previous subtasks
    if status == "incomplete" and loop_count < MAX_LOOPS:
        prev_subtasks = plan.get("subtasks", [])

        # normalize & dedupe simple strings for safe comparison
        def normalize_list(lst):
            return [str(x).strip().lower() for x in (lst or []) if str(x).strip()]

        prev_norm = normalize_list(prev_subtasks)
        new_norm = normalize_list(new_tasks)

        # If the new task list is empty or identical to previous, treat as no-progress -> stop
        if not new_norm or new_norm == prev_norm:
            return {
                "messages": [AIMessage(content="[THINKING]No meaningful plan change detected. Stopping refinement.[/THINKING]")],
                "next_step": "synthesize",
                "loop_count": loop_count,
            }

        # Otherwise, update plan with the new subtasks and increment loop_count
        new_plan = dict(plan)
        new_plan["subtasks"] = new_tasks
        return {
            "messages": [AIMessage(content=f"[THINKING]Gap Analysis: Refining plan ({reasoning})[/THINKING]")],
            "research_plan": new_plan,
            "loop_count": loop_count + 1,
            "next_step": "execute",
        }

    # Otherwise, mark complete (either status==complete or loop cap reached)
    return {
        "messages": [AIMessage(content="[THINKING]Validation Complete.[/THINKING]")],
        "next_step": "synthesize",
        "loop_count": loop_count,
    }


def research_synthesis_node(state: AgentState):
    """RESEARCH MODE: Step 4 - Synthesize"""
    context = get_context(state["user_id"])
    user_goal = state.get("user_goal", "Provide a report.")
    clean_history = get_clean_history(state["messages"], max_messages=20)

    model = ChatOpenAI(model="gpt-4o", temperature=0.2)
    today = datetime.now().strftime("%B %d, %Y")
    sys_msg = SystemMessage(content=f"""
    You are StirFry. Answer the user's request based on the research.
    Current Date: {today}.
    USER CONTEXT: {context}
    STRICT USER GOAL: "{user_goal}"
    INSTRUCTIONS:
    1. If the user asked for a specific format (e.g. "2 sentences", "rating"), YOU MUST OBEY IT.
    2. Prioritize the 'STRICT USER GOAL' above all else.
    3. Ensure all financial data cited is the MOST RECENT available (TTM/2025/2026). Warn if data is old.
    """)
    res = model.invoke([sys_msg] + clean_history)
    return {"messages": [res], "mode": "conversational"}


# --- GRAPH BUILD ---
workflow = StateGraph(AgentState)
workflow.add_node("advisor", advisor_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("research_planner", research_planner_node)
workflow.add_node("research_executor", research_executor_node)
workflow.add_node("research_reflector", research_reflector_node)
workflow.add_node("research_synthesis", research_synthesis_node)


# --- STRICT ENTRY ROUTING ---
def route_entry(state: AgentState):
    mode = state.get("mode", "conversational")
    if mode == "deep_research":
        return "research_planner"
    else:
        return "advisor"


workflow.set_conditional_entry_point(route_entry)


# --- STRICT TOOL ROUTING ---
def route_tool_output(state: AgentState):
    mode = state.get("mode", "conversational")
    if mode == "deep_research":
        return "research_executor"
    return "advisor"


# CASUAL LOOP
workflow.add_conditional_edges("advisor", lambda x: "tools" if x["messages"][-1].tool_calls else END)


# RESEARCH LOOP
workflow.add_edge("research_planner", "research_executor")
workflow.add_conditional_edges("research_executor", lambda x: "tools" if x["messages"][-1].tool_calls else "research_reflector")
workflow.add_conditional_edges("research_reflector", lambda x: "research_executor" if x.get("next_step") == "execute" else "research_synthesis")
workflow.add_edge("research_synthesis", END)

workflow.add_conditional_edges("tools", route_tool_output)


# DB CONNECTION
pool = ConnectionPool(
    conninfo=os.environ["DB_CONNECTION"],
    max_size=10,
    kwargs={"autocommit": True, "sslmode": "require", "prepare_threshold": None, "keepalives": 1, "keepalives_idle": 5},
)
graph = workflow.compile(checkpointer=PostgresSaver(pool))