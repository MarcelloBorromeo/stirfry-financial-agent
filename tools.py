import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool # <--- IMPORT THIS
from supabase import create_client


# LOAD KEYS FIRST
load_dotenv()


# Setup
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
tavily_tool = TavilySearchResults(max_results=3)


# --- NEW TOOL: TRUTH CALCULATOR ---
@tool
def calculate_math(expression: str):
   """
   A precise calculator. Use this for ANY math (sums, net worth, percentages).
   Input should be a simple python math expression string.
   Example: "9465.56 + 2607.43 + 110"
   """
   try:
       # Safe evaluation of math expressions
       allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
       result = eval(expression, {"__builtins__": None}, allowed_names)
       return f"{result}"
   except Exception as e:
       return f"Error calculating: {e}"


# --- EXISTING TOOLS (NOW DECORATED) ---


@tool
def get_market_data(ticker: str):
   """Fetches price, 5-day trend, AND key fundamentals."""
   try:
       stock = yf.Ticker(ticker)
       hist = stock.history(period="5d")
       if hist.empty: return f"Error: No data for {ticker}"
       current = hist['Close'].iloc[-1]
       try:
           info = stock.info
           pe = info.get('trailingPE', 'N/A')
           h52 = info.get('fiftyTwoWeekHigh', 'N/A')
       except: pe, h52 = "N/A", "N/A"
       return f"Ticker: {ticker}\nPrice: ${current:.2f}\nPE: {pe}\n52W High: {h52}\nTrend: {hist['Close'].to_list()}"
   except Exception as e: return f"Error: {e}"


@tool
def update_profile_notes(note: str):
   """Overwrites user profile with new info (Merging logic)."""
   user_id = "marcello_main"
   try:
       res = supabase.table("profiles").select("base_context").eq("id", user_id).maybe_single().execute()
       cur = res.data.get("base_context", "") if res.data else ""
   except: cur = ""
  
   editor = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   prompt = f"Current: {cur}\nNew: {note}\nRewrite profile to include new info. Keep it concise. Output ONLY text."
   new_bio = editor.invoke(prompt).content
  
   supabase.table("profiles").upsert({"id": user_id, "base_context": new_bio}).execute()
   return f"Profile updated: {new_bio}"


@tool
def get_macro_data(dummy: str = "none"):
   """Fetches key US economic indicators from the FRED API."""
   key = os.environ.get("FRED_API_KEY")
   if not key: return "Error: No FRED Key"
   map_ = {"CPI": "CPIAUCSL", "Fed Rate": "FEDFUNDS", "Unemployment": "UNRATE", "10Y-2Y": "T10Y2Y"}
   lines = ["MACRO SNAPSHOT:"]
   for lbl, sid in map_.items():
       try:
           u = f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}&file_type=json&limit=1&sort_order=desc"
           d = requests.get(u, timeout=3).json()
           val = float(d["observations"][0]["value"])
           lines.append(f"- {lbl}: {val}")
       except: lines.append(f"- {lbl}: N/A")
   return "\n".join(lines)


@tool
def get_my_portfolio(dummy: str = "none"):
   """Calculates live value and P&L for the user's portfolio."""
   try:
       res = supabase.table("portfolio").select("*").execute()
       if not res.data: return "Portfolio Empty."
       lines = []
       tot_val = 0
       for i in res.data:
           tik, sh = i['ticker'], float(i['shares'])
           try:
               curr = yf.Ticker(tik).fast_info.last_price
               val = sh * curr
               tot_val += val
               lines.append(f"- {tik}: {sh} sh @ ${curr:.2f} = ${val:.2f}")
           except: lines.append(f"- {tik}: Price Error")
       return f"TOTAL VALUE: ${tot_val:.2f}\n" + "\n".join(lines)
   except Exception as e: return f"Error: {e}"


@tool
def get_company_health(ticker: str):
   """Fetches fundamental financial health metrics using Yahoo Finance."""
   try:
       info = yf.Ticker(ticker).info
       return f"""
       Health ({ticker}):
       - ROE: {info.get('returnOnEquity', 'N/A')}
       - Margins: {info.get('profitMargins', 'N/A')}
       - Debt/Eq: {info.get('debtToEquity', 'N/A')}
       - Rev Growth: {info.get('revenueGrowth', 'N/A')}
       - Free Cash Flow: {info.get('freeCashflow', 'N/A')}
       """
   except Exception as e: return f"Error: {e}"


# --- TOOL LIST ---
tools = [
   calculate_math,
   get_market_data,
   update_profile_notes,
   get_macro_data,
   get_my_portfolio,
   get_company_health,
   tavily_tool
]


# NOW THIS WILL WORK because @tool adds the .name attribute
TOOL_NAME_MAP = {t.name: t for t in tools}



