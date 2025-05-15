import yfinance as yf
from crewai_tools import tool
import requests
import os

from dotenv import load_dotenv

load_dotenv() 
API_KEY = os.getenv("FINNHUB_API_KEY")

def fetch_tickers_in_sector(ticker: str):
    url = f"https://finnhub.io/api/v1/stock/peers?symbol={ticker}&token={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

@tool
def competitor_analysis(ticker: str, num_competitors: int = 3):
    """
    Perform competitor analysis for a given stock.
    
    Args:
        ticker (str): The stock ticker symbol.
        num_competitors (int): Number of top competitors to analyze.
    
    Returns:
        dict: Competitor analysis results.
    """
    print(f"[TOOL CALL] competitor_analysis used with ticker={ticker}")
    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get('sector')
    industry = info.get('industry')
    
    #Eredeti  Get competitors in the same industry
    #industry_stocks = yf.Ticker(f"^{sector}").info.get('components', [])
    #competitors = [comp for comp in industry_stocks if comp != ticker][:num_competitors]
    
    """ competitor_data = []
    for comp in competitors:
        comp_stock = yf.Ticker(comp)
        comp_info = comp_stock.info
        competitor_data.append({
            "ticker": comp,
            "name": comp_info.get('longName'),
            "market_cap": comp_info.get('marketCap'),
            "pe_ratio": comp_info.get('trailingPE'),
            "revenue_growth": comp_info.get('revenueGrowth'),
            "profit_margins": comp_info.get('profitMargins')
        }) """
    
    related_tickers = fetch_tickers_in_sector(ticker)
    related_tickers = [comp for comp in related_tickers if comp != ticker][:num_competitors]

    
    competitor_data = []
    for comp in related_tickers:
        try:
            comp_stock = yf.Ticker(comp)
            comp_info = comp_stock.info
            competitor_data.append({
                "ticker": comp,
                "name": comp_info.get("longName", "N/A"),
                "market_cap": comp_info.get("marketCap", "N/A"),
                "pe_ratio": comp_info.get("trailingPE", "N/A"),
                "revenue_growth": comp_info.get("revenueGrowth", "N/A"),
                "profit_margins": comp_info.get("profitMargins", "N/A")
            })
        except Exception as e:
            print(f"Error fetching data for competitor {comp}: {e}")
            continue

    return {
        "main_stock": ticker,
        "industry": industry,
        "competitors": competitor_data
    }

