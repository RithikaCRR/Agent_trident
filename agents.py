"""
AI Trading Agents
Three specialized agents + a moderator that debate to produce trading decisions.
Uses Requesty.ai (OpenAI-compatible API).
"""

import json
import requests
from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

AI_API_KEY = os.getenv("AI_API_KEY")
AI_BASE_URL = os.getenv("AI_BASE_URL")
MODEL = os.getenv("MODEL")

# Initialize OpenAI client with Requesty/Featherless config
client = OpenAI(
    api_key=AI_API_KEY,
    base_url=AI_BASE_URL
)


def _call_ai(system_prompt: str, user_prompt: str) -> str:
    """Call AI chat completions API using the OpenAI SDK."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[agents] AI call error: {e}")
        return json.dumps({
            "action": "HOLD",
            "confidence": 0,
            "reasoning": f"AI service error: {str(e)}"
        })



def _parse_agent_response(raw: str) -> dict:
    """Extract JSON from agent response (handles markdown code blocks)."""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON-like content
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {
            "action": "HOLD",
            "confidence": 50,
            "reasoning": raw[:500]
        }


def run_technical_agent(symbol: str, stock_data: dict, indicators: dict) -> dict:
    """Technical Analysis Agent — focuses on price patterns, indicators, and volume."""
    system_prompt = """You are a Technical Analysis Trading Agent. You analyze stock charts, 
price patterns, moving averages, RSI, volume, and momentum indicators.

You must respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
    "action": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "reasoning": "Your detailed technical analysis reasoning",
    "key_signals": ["signal1", "signal2"]
}"""

    user_prompt = f"""Analyze {symbol} using the following technical data:

Current Quote: {json.dumps(stock_data, indent=2)}

Technical Indicators: {json.dumps(indicators, indent=2)}

Provide your trading recommendation based purely on technical analysis."""

    raw = _call_ai(system_prompt, user_prompt)
    result = _parse_agent_response(raw)
    result["agent"] = "Technical Analyst"
    return result


def run_sentiment_agent(symbol: str, stock_data: dict, market_overview: dict) -> dict:
    """Sentiment Analysis Agent — focuses on market mood, news, and sentiment."""
    system_prompt = """You are a Market Sentiment Trading Agent. You analyze market mood, 
news sentiment, sector trends, and overall market conditions.

You must respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
    "action": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "reasoning": "Your detailed sentiment analysis reasoning",
    "key_signals": ["signal1", "signal2"]
}"""

    user_prompt = f"""Analyze the sentiment and market conditions for {symbol}:

Current Stock Data: {json.dumps(stock_data, indent=2)}

Market Overview (Top Gainers/Losers/Active):
{json.dumps(market_overview, indent=2)}

The current date context: This is real-time market data.
Assess the market sentiment and momentum for this stock."""

    raw = _call_ai(system_prompt, user_prompt)
    result = _parse_agent_response(raw)
    result["agent"] = "Sentiment Analyst"
    return result


def run_risk_agent(symbol: str, stock_data: dict, indicators: dict,
                   technical_opinion: dict, sentiment_opinion: dict,
                   balance: float) -> dict:
    """Risk Management Agent — evaluates risk/reward, position sizing, stop-losses."""
    system_prompt = """You are the Risk Management Trading Agent and final decision maker. You evaluate risk/reward ratios, 
position sizing, portfolio exposure, stop-loss levels, downside protection, and synthesize the opinions of the Technical and Sentiment agents to make the FINAL trading decision.

You must respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
    "action": "BUY" or "SELL" or "HOLD",
    "confidence": 0-100,
    "reasoning": "Your detailed risk analysis reasoning",
    "key_signals": ["signal1", "signal2"],
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "suggested_position_pct": 0-100,
    "stop_loss_pct": 0-100,
    "summary": "Brief summary of the consensus and your final decision",
    "dissenting_views": "Any notable disagreements between the other agents"
}"""

    user_prompt = f"""Evaluate the risk for trading {symbol}:

Stock Data: {json.dumps(stock_data, indent=2)}
Technical Indicators: {json.dumps(indicators, indent=2)}

Technical Agent Opinion: {json.dumps(technical_opinion, indent=2)}
Sentiment Agent Opinion: {json.dumps(sentiment_opinion, indent=2)}

Current Portfolio Balance: ${balance:,.2f}

Assess the risk/reward and provide your recommendation."""

    raw = _call_ai(system_prompt, user_prompt)
    result = _parse_agent_response(raw)
    result["agent"] = "Risk Manager"
    return result



def run_full_analysis(symbol: str, stock_data: dict, indicators: dict,
                      market_overview: dict, balance: float) -> dict:
    """Run the complete multi-agent analysis pipeline."""
    print(f"[agents] Starting analysis for {symbol}...")

    # 1. Technical Agent
    print("[agents] Running Technical Agent...")
    technical = run_technical_agent(symbol, stock_data, indicators)
    print(f"[agents] Technical: {technical.get('action')} (conf: {technical.get('confidence')})")

    # 2. Sentiment Agent
    print("[agents] Running Sentiment Agent...")
    sentiment = run_sentiment_agent(symbol, stock_data, market_overview)
    print(f"[agents] Sentiment: {sentiment.get('action')} (conf: {sentiment.get('confidence')})")

    # 3. Risk Agent (sees others' opinions)
    print("[agents] Running Risk Management Agent...")
    risk = run_risk_agent(symbol, stock_data, indicators, technical, sentiment, balance)
    print(f"[agents] Risk: {risk.get('action')} (conf: {risk.get('confidence')})")

    return {
        "symbol": symbol,
        "agents": {
            "technical": technical,
            "sentiment": sentiment,
            "risk": risk,
        },
        "final_decision": {
            "action": risk.get("action", "HOLD"),
            "confidence": risk.get("confidence", 0),
            "summary": risk.get("summary", ""),
            "dissenting_views": risk.get("dissenting_views", ""),
        },
    }

if __name__ == "__main__":
    import stock_data
    symbol = "bitcoin"
    print(f"--- Testing agents.py with {symbol} ---")
    
    # Fetch real data for test
    quote = stock_data.get_quote(symbol) or {"symbol": symbol, "price": 65000}
    intraday = stock_data.get_intraday(symbol, "5min")
    indicators = stock_data.compute_technical_indicators(intraday) if intraday else {}
    market_overview = stock_data.get_top_gainers_losers()
    
    # Run analysis
    result = run_full_analysis(symbol, quote, indicators, market_overview, balance=10000.0)
    
    print("\n--- Final Output ---")
    print(json.dumps(result, indent=2))

