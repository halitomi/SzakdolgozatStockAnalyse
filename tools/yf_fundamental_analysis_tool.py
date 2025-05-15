from dotenv import load_dotenv
import yfinance as yf
from crewai_tools import tool
import finnhub
import requests
import os

load_dotenv() 
API_KEY = os.getenv("FINNHUB_API_KEY")

@tool
def yf_fundamental_analysis(ticker: str):
    """
    Perform comprehensive fundamental analysis on a given stock ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
    
    Returns:
        dict: Comprehensive fundamental analysis results.
    """
    print(f"[TOOL CALL] yf_fundamental_analysis used with ticker={ticker}")
    finnhub_client = finnhub.Client(api_key=API_KEY)
    finnhub_metrics = {}
    try:
        url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        finnhub_metrics = response.json().get("metric", {})
    except Exception as e:
        print(f"Error fetching Finnhub Metrics: {e}")

    """ url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        metrics = response.json().get("metric", {})
        print("Finnhub Metrics:", metrics)
        current_ratio = metrics.get("currentRatio")
        debt_to_equity = metrics.get("debtEquityRatio")
        roe = metrics.get("roeTTM") """

    stock = yf.Ticker(ticker)
    info = stock.info
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow
    
    # Calculate additional financial ratios
    stock = yf.Ticker(ticker)
    info = stock.info
    financials = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow = stock.cashflow

    # Combine data from Finnhub and YFinance
    try:
        current_ratio = (
            finnhub_metrics.get("currentRatioAnnual")
            or (balance_sheet.loc['Total Current Assets'].iloc[-1]
                / balance_sheet.loc['Total Current Liabilities'].iloc[-1]
                if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index
                else None)
        )
        debt_to_equity = (
            finnhub_metrics.get("totalDebt/totalEquityAnnual")
            or (balance_sheet.loc['Total Liabilities'].iloc[-1]
                / balance_sheet.loc['Total Stockholder Equity'].iloc[-1]
                if 'Total Liabilities' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index
                else None)
        )
        roe = (
            finnhub_metrics.get("roeTTM")
            or (financials.loc['Net Income'].iloc[-1]
                / balance_sheet.loc['Total Stockholder Equity'].iloc[-1]
                if 'Net Income' in financials.index and 'Total Stockholder Equity' in balance_sheet.index
                else None)
        )
        roa = (
            finnhub_metrics.get("roaTTM")
            or (financials.loc['Net Income'].iloc[-1]
                / balance_sheet.loc['Total Assets'].iloc[-1]
                if 'Net Income' in financials.index and 'Total Assets' in balance_sheet.index
                else None)
        )
        revenue_growth = (
            finnhub_metrics.get("revenueGrowthTTMYoy")
            or ((
                financials.loc['Total Revenue'].iloc[-1] - financials.loc['Total Revenue'].iloc[-2]
            ) / financials.loc['Total Revenue'].iloc[-2]
                if 'Total Revenue' in financials.index and len(financials.loc['Total Revenue']) > 1
                else None)
        )
        net_income_growth = (
            finnhub_metrics.get("netIncomeGrowthTTMYoy")
            or ((
                financials.loc['Net Income'].iloc[-1] - financials.loc['Net Income'].iloc[-2]
            ) / financials.loc['Net Income'].iloc[-2]
                if 'Net Income' in financials.index and len(financials.loc['Net Income']) > 1
                else None)
        )
        fcf = (
            (cash_flow.loc['Operating Cash Flow'].iloc[-1]
             - cash_flow.loc['Capital Expenditures'].iloc[-1]
             if 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditures' in cash_flow.index
             else None)
        )
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        current_ratio = debt_to_equity = roe = roa = revenue_growth = net_income_growth = fcf = None

    # Return Comprehensive Analysis
    return {
        "ticker": ticker,
        "company_name": info.get('longName'),
        "sector": info.get('sector'),
        "industry": info.get('industry'),
        "market_cap": info.get('marketCap'),
        "pe_ratio": info.get('trailingPE'),
        "forward_pe": info.get('forwardPE'),
        "peg_ratio": info.get('pegRatio'),
        "price_to_book": info.get('priceToBook'),
        "dividend_yield": info.get('dividendYield'),
        "beta": info.get('beta'),
        "52_week_high": info.get('fiftyTwoWeekHigh'),
        "52_week_low": info.get('fiftyTwoWeekLow'),
        "current_ratio": current_ratio,
        "debt_to_equity": debt_to_equity,
        "return_on_equity": roe,
        "return_on_assets": roa,
        "revenue_growth": revenue_growth,
        "net_income_growth": net_income_growth,
        "free_cash_flow": fcf,
        "analyst_recommendation": info.get('recommendationKey'),
        "target_price": info.get('targetMeanPrice'),
    }