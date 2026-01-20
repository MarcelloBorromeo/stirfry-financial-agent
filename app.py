import os
import base64
import re
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client
from langchain_core.messages import HumanMessage, AIMessage
from graph import graph


load_dotenv()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
st.set_page_config(page_title="StirFry", layout="wide", page_icon="🍳")


# --- CUSTOM AVATAR ENGINE ---
def get_icon(role):
   if role == "user":
       svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="50" fill="#D8A49B"/><circle cx="50" cy="35" r="18" fill="white"/><path d="M20 85 Q20 60 50 60 Q80 60 80 85 Z" fill="white"/></svg>"""
   else:
       svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="50" fill="#88A0A8"/><rect x="25" y="35" width="50" height="35" rx="5" fill="white"/><circle cx="50" cy="52" r="8" fill="#88A0A8"/><rect x="40" y="60" width="20" height="3" fill="#88A0A8"/></svg>"""
   b64 = base64.b64encode(svg.encode()).decode()
   return f"data:image/svg+xml;base64,{b64}"


# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;700&family=Press+Start+2P&display=swap');
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }


html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.main .block-container { padding-bottom: 200px !important; }


.footer-status-container {
  position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
  width: 90%; max-width: 700px; z-index: 9999; display: flex; flex-direction: column; gap: 10px; pointer-events: none;
}
.footer-status-container > * { pointer-events: auto; }


[data-testid="stStatusWidget"] { background-color: #1E1E1E !important; border: 1px solid #546A76 !important; border-radius: 12px !important; }
[data-testid="stStatusWidget"] svg, [data-testid="stStatusWidget"] label, [data-testid="stStatusWidget"] p { color: #88A0A8 !important; }


.step-card {
  display: flex; align-items: center; gap: 15px; background-color: #2b3b45;
  border: 1px solid #B4CEB3; border-left: 6px solid #B4CEB3; border-radius: 8px;
  padding: 16px; animation: fadeUp 0.3s ease-out forwards; width: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.custom-spinner {
  width: 24px; height: 24px; border: 3px solid rgba(180, 206, 179, 0.2);
  border-top: 3px solid #B4CEB3; border-radius: 50%; animation: spin 1s linear infinite; flex-shrink: 0;
}
.step-text { color: #B4CEB3; font-weight: 600; font-family: 'Space Grotesk', monospace; }


.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #546A76; color: white; border-radius: 12px; border: 1px solid #3E4E59; }
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #262730; border: 1px solid #88A0A8; border-radius: 12px; }
.big-title { font-family: 'Press Start 2P', monospace; font-size: 4rem; color: #B4CEB3; margin-bottom: 10px; }
.subtitle { font-size: 1.5rem; color: #B4CEB3; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
if "mode" not in st.session_state: st.session_state.mode = "conversational"
st.markdown('<div class="big-title">StirFry.</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Marcello\'s stateful financial researcher.</div>', unsafe_allow_html=True)


# --- HELPERS ---
def get_market_price(ticker):
  try:
      return float(yf.Ticker(ticker).fast_info.last_price)
  except: return None


def fetch_position(user_id, ticker):
  try:
      resp = supabase.table("portfolio").select("shares,avg_cost").eq("user_id", user_id).eq("ticker", ticker).maybe_single().execute()
      return (float(resp.data.get("shares", 0)), float(resp.data.get("avg_cost", 0))) if resp.data else (0.0, 0.0)
  except: return 0.0, 0.0


def upsert_portfolio(user_id, ticker, shares, avg_cost):
  supabase.table("portfolio").upsert({ "user_id": user_id, "ticker": ticker, "shares": float(shares), "avg_cost": float(avg_cost) }, on_conflict="user_id, ticker").execute()


def render_active_step(text, warning=False):
   col = "#FFC107" if warning else "#B4CEB3"
   return f"""<div class="step-card" style="border-left-color:{col}"><div class="custom-spinner" style="border-top-color:{col}"></div><div class="step-text" style="color:{col}">{text}</div></div>"""


# --- SIDEBAR ---
with st.sidebar:
  st.header("Settings")
  user_id = st.text_input("User ID", value="marcello_main")
  st.divider()
  mode = st.radio("Operating Mode", ["Geeked", "Locked [Research]"], index=1 if st.session_state.mode == "deep_research" else 0)
  st.session_state.mode = "deep_research" if mode == "Locked [Research]" else "conversational"
  st.divider()
 
  # Quick Update Form
  st.subheader("Quick Update")
  with st.form("trade_form", clear_on_submit=True):
      c1, c2 = st.columns(2)
      action = c1.selectbox("Action", ["Buy", "Sell"])
      qty = c1.number_input("Shares", min_value=0.0, step=1.0)
      ticker = c2.text_input("Ticker").upper().strip()
      manual_price = c2.number_input("Price ($)", min_value=0.0)
      if st.form_submit_button("Execute"):
          price = manual_price or get_market_price(ticker)
          if price and ticker and qty > 0:
              old_s, old_c = fetch_position(user_id, ticker)
              if action == "Buy":
                  new_s, new_avg = old_s + qty, ((old_s * old_c) + (qty * price)) / (old_s + qty)
              else:
                  new_s, new_avg = max(0, old_s - qty), old_c
              upsert_portfolio(user_id, ticker, new_s, new_avg)
              st.success(f"Executed: {action} {qty} {ticker}")


# --- CHAT ENGINE ---
if "messages" not in st.session_state: st.session_state.messages = []


for msg in st.session_state.messages:
   if msg.content:
       clean = re.sub(r'\[THINKING\].*?\[/THINKING\]', '', msg.content, flags=re.DOTALL).strip()
       if clean:
           with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant", avatar=get_icon("user" if isinstance(msg, HumanMessage) else "assistant")):
               st.markdown(clean)


if prompt := st.chat_input("Ask StirFry..."):
  if st.session_state.mode == "deep_research" and not prompt.startswith("DEEP_RESEARCH:"):
      prompt = f"DEEP_RESEARCH: {prompt}"
 
  st.session_state.messages.append(HumanMessage(content=prompt))
  with st.chat_message("user", avatar=get_icon("user")): st.markdown(prompt)


  with st.chat_message("assistant", avatar=get_icon("assistant")):
      response_placeholder = st.empty()
      full_response = ""
      active_mode = "advisor"


      # --- FIXED FOOTER UI ---
      with st.container():
          st.markdown('<div class="footer-status-container">', unsafe_allow_html=True)
          active_step_placeholder = st.empty()
         
          with st.status("System Logs", expanded=False) as log_status:
             
              # CONFIGURATION WITH RECURSION LIMIT FIX
              config = {
                  "configurable": {"thread_id": user_id, "user_id": user_id},
                  "recursion_limit": 150  # <--- THIS IS THE CRITICAL FIX
              }
             
              state = {"messages": st.session_state.messages, "user_id": user_id, "mode": st.session_state.mode, "research_session_id": None, "research_plan": None, "loop_count": 0}
             
              for chunk, meta in graph.stream(state, config, stream_mode="messages"):
                  node = meta.get("langgraph_node", "")
                 
                  # Reset Logic for Research Mode
                  if node == "research_planner" and active_mode == "advisor":
                      full_response = ""
                      response_placeholder.empty()
                      active_mode = "research"


                  # Banner Updates
                  if node == "research_planner" and "[THINKING]" in chunk.content:
                      active_step_placeholder.markdown(render_active_step("Blueprint Created."), unsafe_allow_html=True)
                 
                  if node == "research_reflector":
                      if "Gap Analysis" in chunk.content:
                          active_step_placeholder.markdown(render_active_step("Gaps Found. Refining...", warning=True), unsafe_allow_html=True)
                          st.warning(chunk.content.replace("[THINKING]", "").replace("[/THINKING]", ""))
                      else:
                          active_step_placeholder.markdown(render_active_step("Validation Complete."), unsafe_allow_html=True)


                  if isinstance(chunk, AIMessage) and chunk.tool_calls:
                      nm = chunk.tool_calls[0]['name']
                      active_step_placeholder.markdown(render_active_step(f"Accessing {nm}..."), unsafe_allow_html=True)
                      st.write(f"⚙️ **Invoking:** `{nm}`")


                  # Stream Output (Gatekeeper)
                  if isinstance(chunk, AIMessage) and chunk.content:
                      if node == "advisor" and active_mode == "advisor":
                          full_response += chunk.content
                          response_placeholder.markdown(full_response + "▌")
                      elif node == "research_synthesis":
                          if not full_response:
                              active_step_placeholder.markdown(render_active_step("Synthesizing Report..."), unsafe_allow_html=True)
                          full_response += chunk.content
                          response_placeholder.markdown(full_response + "▌")


              log_status.update(label="System Logs (Complete)", state="complete", expanded=False)
              active_step_placeholder.empty()
          st.markdown('</div>', unsafe_allow_html=True)


  if full_response:
      st.session_state.messages.append(AIMessage(content=full_response))





