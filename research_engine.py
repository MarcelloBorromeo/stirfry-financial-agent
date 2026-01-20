import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum


from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# --- DATA MODELS ---


class SourceType(Enum):
   FINANCIAL_DATA = "financial_data"
   NEWS = "news"
   REGULATORY = "regulatory"
   MARKET_DATA = "market_data"
   ACADEMIC = "academic"
   COMPANY_FILING = "company_filing"


@dataclass
class ToolResult:
   id: str
   tool_name: str
   query: str
   raw_output: str
   summary: str
   timestamp: str
   source_type: SourceType
   metadata: Dict[str, Any] = field(default_factory=dict)
  
   def to_citation(self) -> str:
       date = self.timestamp.split('T')[0]
       return f"[{self.tool_name} | {date} | {self.id}]"


@dataclass
class Evidence:
   claim: str
   tool_results: List[ToolResult]
   confidence: float
   verification_notes: str
  
   def get_citations(self) -> List[str]:
       return [tr.to_citation() for tr in self.tool_results]


@dataclass
class ThesisPoint:
   point: str
   evidence: List[Evidence]
   confidence: float
   risks: List[str] = field(default_factory=list)
   open_questions: List[str] = field(default_factory=list)


@dataclass
class ResearchPlan:
   objective: str
   subtasks: List[str]
   tools_to_use: List[str]
   expected_outputs: List[str]
   stop_criteria: str
   estimated_tool_calls: int = 0


@dataclass
class ResearchSession:
   session_id: str
   objective: str
   plan: Optional[ResearchPlan] = None
   tool_results: List[ToolResult] = field(default_factory=list)
   thesis_points: List[ThesisPoint] = field(default_factory=list)
   methodology: str = ""
   limitations: List[str] = field(default_factory=list)
   started_at: str = field(default_factory=lambda: datetime.now().isoformat())
   completed_at: Optional[str] = None


class ResearchMemory:
   def __init__(self):
       self.sessions: Dict[str, ResearchSession] = {}
       self.tool_results_index: Dict[str, List[ToolResult]] = {}
  
   def create_session(self, session_id: str, objective: str) -> ResearchSession:
       session = ResearchSession(session_id=session_id, objective=objective)
       self.sessions[session_id] = session
       return session
  
   def save_tool_result(self, session_id: str, result: ToolResult):
       if session_id in self.sessions:
           self.sessions[session_id].tool_results.append(result)
       query_key = f"{result.tool_name}:{result.query}"
       if query_key not in self.tool_results_index:
           self.tool_results_index[query_key] = []
       self.tool_results_index[query_key].append(result)
  
   def get_by_query(self, tool_name: str, query: str) -> List[ToolResult]:
       query_key = f"{tool_name}:{query}"
       return self.tool_results_index.get(query_key, [])
  
   def get_session(self, session_id: str) -> Optional[ResearchSession]:
       return self.sessions.get(session_id)


# --- COGNITIVE MODULES ---


class ResearchPlanner:
   def __init__(self, model_name: str = "gpt-4o-mini"):
       self.model = ChatOpenAI(model=model_name, temperature=0)
  
   def create_plan(self, objective: str, available_tools: List[str]) -> ResearchPlan:
       system_prompt = """You are a research planning expert.
Available tools: get_market_data, get_company_health, get_macro_data, tavily_search, get_my_portfolio.
Return ONLY valid JSON:
{
   "objective": "clear goal",
   "subtasks": ["step 1", "step 2"],
   "tools_to_use": ["tool1", "tool2"],
   "expected_outputs": ["output1"],
   "stop_criteria": "when done",
   "estimated_tool_calls": 5
}"""
       user_prompt = f"Objective: {objective}\nTools: {available_tools}"


       try:
           response = self.model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
           content = response.content.strip()
           if content.startswith('```'):
               content = content.split('```')[1].replace('json', '').strip()
           plan_dict = json.loads(content)
          
           tools = plan_dict.get('tools_to_use', [])
           if tools and isinstance(tools[0], list):
               tools = [item for sublist in tools for item in sublist]
           plan_dict['tools_to_use'] = [str(t) for t in tools if t]
          
           if 'estimated_tool_calls' not in plan_dict:
               plan_dict['estimated_tool_calls'] = len(plan_dict.get('subtasks', [])) * 2
          
           return ResearchPlan(**plan_dict)
       except:
           return ResearchPlan(
               objective=objective,
               subtasks=["Gather data", "Analyze"],
               tools_to_use=["get_market_data", "tavily_search"],
               expected_outputs=["Analysis"],
               stop_criteria="Done",
               estimated_tool_calls=5
           )


class Verifier:
   def __init__(self, model_name: str = "gpt-4o-mini"):
       self.model = ChatOpenAI(model=model_name, temperature=0)
  
   def verify_claim(self, claim: str, tool_results: List[ToolResult]) -> Dict[str, Any]:
       evidence_text = "\n".join([f"{tr.tool_name}: {tr.summary[:200]}" for tr in tool_results])
       system_prompt = """Verify the claim based on evidence. Return JSON:
       {"verified": bool, "confidence_score": float, "supporting_points": [], "contradictions": [], "notes": ""}"""
      
       try:
           response = self.model.invoke([
               SystemMessage(content=system_prompt),
               HumanMessage(content=f"CLAIM: {claim}\nEVIDENCE: {evidence_text}")
           ])
           content = response.content.strip()
           if content.startswith('```'): content = content.split('```')[1].replace('json', '').strip()
           return json.loads(content)
       except:
           return {"verified": False, "confidence_score": 0.0, "notes": "Verification failed"}


class ResearchSynthesizer:
   def __init__(self, model_name: str = "gpt-4o-mini"):
       self.model = ChatOpenAI(model=model_name, temperature=0.2)
  
   def synthesize(self, objective: str, verified_findings: List[Dict[str, Any]], tool_results: List[ToolResult]) -> List[ThesisPoint]:
       findings_text = str(verified_findings)[:4000]
       system_prompt = """Synthesize findings into investment thesis points.
       Return JSON array of objects with keys: point, evidence (list of objects), confidence, risks, open_questions."""
      
       try:
           response = self.model.invoke([
               SystemMessage(content=system_prompt),
               HumanMessage(content=f"Obj: {objective}\nFindings: {findings_text}")
           ])
           content = response.content.strip()
           if content.startswith('```'): content = content.split('```')[1].replace('json', '').strip()
           data = json.loads(content)
          
           points = []
           for d in data:
               points.append(ThesisPoint(
                   point=d['point'],
                   evidence=[],
                   confidence=d.get('confidence', 0.5),
                   risks=d.get('risks', []),
                   open_questions=d.get('open_questions', [])
               ))
           return points
       except:
           return [ThesisPoint(point="Synthesis failed", evidence=[], confidence=0.0)]


# --- NEW: THE CRITIC (REFLECTOR) ---
class ResearchReflector:
   """Reviews gathered evidence and identifies gaps."""
  
   def __init__(self, model_name: str = "gpt-4o", temperature=0):
       self.model = ChatOpenAI(model=model_name, temperature=temperature)


   def review_findings(self, objective: str, tool_results: List[ToolResult]) -> Dict[str, Any]:
       """
       Analyzes current results against the objective.
       Returns a decision to 'continue' or 'finish'.
       """
       evidence_summary = "\n".join([
           f"- Source: {tr.tool_name} | Summary: {tr.summary[:300]}..."
           for tr in tool_results
       ])
      
       system_prompt = """You are a Senior Research Editor.
       Review the evidence gathered so far against the Research Objective.
      
       DECISION CRITERIA:
       1. MISSING DATA: Are there specific numbers/facts requested that are missing?
       2. VAGUENESS: Are the findings too generic?
       3. CONFLICT: Is there contradictory data that needs resolution?
      
       Output a JSON object:
       {
           "status": "complete" OR "incomplete",
           "reasoning": "Explanation of why...",
           "follow_up_tasks": ["precise list of 1-3 new search queries or tool calls needed"]
       }
      
       If status is "complete", follow_up_tasks must be empty.
       Do not nitpick. Only request more data if the core objective is unmet.
       """


       user_prompt = f"OBJECTIVE: {objective}\nEVIDENCE GATHERED:\n{evidence_summary}"
      
       try:
           response = self.model.invoke([
               SystemMessage(content=system_prompt),
               HumanMessage(content=user_prompt)
           ])
           content = response.content.strip()
           if content.startswith('```'):
               content = content.split('```')[1].replace('json', '').strip()
           return json.loads(content)
       except Exception as e:
           return {"status": "complete", "reasoning": f"Error: {e}", "follow_up_tasks": []}


class DeepResearchEngine:
   def __init__(self):
       self.memory = ResearchMemory()
       self.planner = ResearchPlanner()
       self.verifier = Verifier()
       self.synthesizer = ResearchSynthesizer()
       self.reflector = ResearchReflector() # <--- REGISTER REFLECTOR


   def execute_research(self, session_id: str, objective: str, tool_executor: Any, available_tools: List[str]):
       session = self.memory.create_session(session_id, objective)
       plan = self.planner.create_plan(objective, available_tools)
       session.plan = plan
       return {"session_id": session_id, "plan": asdict(plan)}


   def verify_and_synthesize(self, session_id: str):
       session = self.memory.get_session(session_id)
       if not session: return {"error": "Session not found"}
      
       verified = []
       for res in session.tool_results:
           verified.append(self.verifier.verify_claim(res.summary, [res]))
          
       thesis = self.synthesizer.synthesize(session.objective, verified, session.tool_results)
       session.thesis_points = thesis
       session.completed_at = datetime.now().isoformat()
      
       return self._build_report(session)


   def _build_report(self, session):
       return {
           "session_id": session.session_id,
           "objective": session.objective,
           "executive_summary": "Research Complete",
           "thesis_points": [asdict(tp) for tp in session.thesis_points],
           "methodology": "Deep Research",
           "limitations": [],
           "completed_at": session.completed_at
       }
