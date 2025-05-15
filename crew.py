from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import tool
#from langchain_community.llms import Ollama
#from langchain_ollama import OllamaLLM
#from langchain_ollama import OllamaLLM
from llm_config import init_llm
from tools.yf_tech_analysis_tool import yf_tech_analysis
from tools.yf_fundamental_analysis_tool import yf_fundamental_analysis
from tools.sentiment_analysis_tool import sentiment_analysis
from tools.competitor_analysis_tool import competitor_analysis
from tools.risk_assessment_tool import risk_assessment
from llm_config import init_llm

""" from Multi_Agent.tools.yf_tech_analysis_tool import yf_tech_analysis
from Multi_Agent.tools.yf_fundamental_analysis_tool import yf_fundamental_analysis
from Multi_Agent.tools.sentiment_analysis_tool2 import sentiment_analysis
from Multi_Agent.tools.competitor_analysis_tool import competitor_analysis
from Multi_Agent.tools.risk_assessment_tool import risk_assessment """

import os
import ollama
import requests
os.environ['OLLAMA_API_KEY'] = ''
os.environ['OLLAMA_API_BASE'] = 'http://localhost:11434'



def generate_indicator_description(indicator):
    # Define the base URL for Ollama
    base_url = "http://localhost:11434/api/completions"
    
    # Create the payload with the model and prompt
    payload = {
        "model": "llama3.2",
        "prompt": f"""
        Provide a detailed explanation of the stock indicator '{indicator}'. 
        Include:
        - What the indicator measures
        - How it is calculated
        - How traders can interpret it
        """,
        "temperature": 0.7
    }

    try:
        # Make a POST request to Ollama's API
        response = requests.post(base_url, json=payload)
        response.raise_for_status()  # Check for HTTP errors

        # Parse the response and return the text
        result = response.json()
        return result.get("completion", "No description returned.")
    except requests.RequestException as e:
        return f"Error connecting to Ollama: {e}"
    except Exception as e:
        return f"Error generating description: {e}"
    

indicator_analyst = Agent(
    role="Financial Indicator Analyst",
    goal="Provide detailed explanations and analysis for financial indicators.",
    backstory="You are an expert in financial analysis, specializing in technical indicators.",
    tools=[],
    memory=False,
    verbose=False
)

# Define the task
analyze_task = Task(
    description=(
        "Analyze the {indicator} indicator for the stock {ticker}. "
        "Explain its meaning, calculation, and what it implies about the stock's performance."
    ),
    expected_output="A clear and concise explanation of the indicator for stock {ticker}.",
    agent=indicator_analyst
)

# Create the Crew
crew = Crew(
    agents=[indicator_analyst],
    tasks=[analyze_task],
    process=Process.sequential
)

#def create_crew(stock_symbol, model_name="ollama/llama3.2", provider="local"):

def create_crew(stock_symbol, model_name, provider):
    llm = init_llm(model_name, provider)
    
    print("LLM initialized successfully.", llm)
    #llm = LLM(
    #model='ollama/llama3.2',
    #base_url='http://localhost:11434',
    #temperature=0.7,
    #verbose=True
    #)   

    #try:
        #test_response = llm.invoke("Hello, can you respond?")
        #print("LLM Test Response:", test_response)
    #except Exception as e:
        #print(f"LLM Initialization Error: {e}")
    #print("LLM initialized:", llm)

    # Define Agents
    researcher = Agent(
        role='Stock Market Researcher',
        #goal='Gather and analyze comprehensive data about the stock',
        goal='Analyze and interpret financial and technical data to uncover trends, opportunities, and risks for the stock.',
        #backstory="You're an experienced stock market researcher with a keen eye for detail and a talent for uncovering hidden trends.",
        backstory="You're an experienced stock market researcher with a deep understanding of fundamental and technical analysis. You excel in identifying key patterns and providing actionable insights.",
        tools=[yf_tech_analysis, yf_fundamental_analysis, competitor_analysis],
        llm=llm
    )

    analyst = Agent(
        role='Financial Analyst',
        goal='Analyze the gathered data and provide investment insights',
        backstory="You're a seasoned financial analyst known for your accurate predictions and ability to synthesize complex information.",
        tools=[yf_tech_analysis, yf_fundamental_analysis, risk_assessment],
        llm=llm
    )

    sentiment_analyst = Agent(
        role='Sentiment Analyst',
        goal='Analyze market sentiment and its potential impact on the stock',
        backstory="You're an expert in behavioral finance and sentiment analysis, capable of gauging market emotions and their effects on stock performance.",
        tools=[sentiment_analysis],
        llm=llm
    )

    strategist = Agent(
        role='Investment Strategist',
        goal='Develop a comprehensive investment strategy based on all available data',
        backstory="You're a renowned investment strategist known for creating tailored investment plans that balance risk and reward.",
        tools=[],
        llm=llm
    )

    # Define Tasks
    # """research_task = Task(
    #     description=f"Research {stock_symbol} using advanced technical and fundamental analysis tools. Provide a comprehensive summary of key metrics, including chart patterns, financial ratios, and competitor analysis.",
    #     agent=researcher,
    #     expected_output="A detailed report covering technical and fundamental analysis, and competitor performance."
    # )"""

    research_task = Task(
    description=(
        f"""You are a professional financial analyst. Use all of your tool to get technical indicator data for the stock {stock_symbol}. Your task:
- Interpret the key technical indicators such as RSI, MACD, ATR (volatility), Support, and Resistance levels.
- Explain whether the stock is showing bullish (uptrend) or bearish (downtrend) signals.
- Comment on the current volatility.
- Summarize how technical factors could impact investment decisions.


Deliver your findings in the form of a clear, concise, and well-structured technical analysis report. Make sure your explanation is accessible to non-experts but valuable to experienced investors. Provide actionable insights wherever possible.
"""
    ),
    expected_output="A full human-readable technical analysis report.",
    agent=researcher
)

    sentiment_task = Task(
        description=f"Use your tool and analyze the market sentiment for {stock_symbol} using news and social media data. Evaluate how current sentiment might affect the stock's performance.",
        agent=sentiment_analyst,
        expected_output="A sentiment analysis summary highlighting positive, negative, and neutral sentiment trends."
    )

    analysis_task = Task(
        description=f"Use your tools to synthesize the research data and sentiment analysis for {stock_symbol}. Conduct a thorough risk assessment and provide a detailed analysis of the stock's potential.",
        agent=analyst,
        expected_output="A risk assessment report with detailed analysis of potential risks and opportunities."
    )

    strategy_task = Task(
        description=f"Based on all the gathered information about {stock_symbol}, develop a comprehensive investment strategy. Consider various scenarios and provide actionable recommendations for different investor profiles.",
        agent=strategist,
        expected_output="An investment strategy tailored for different investor profiles and risk levels."
    )

    # Create Crew
    crew = Crew(
        agents=[researcher, sentiment_analyst, analyst, strategist],
        tasks=[research_task, sentiment_task, analysis_task, strategy_task],
        process=Process.sequential
    )

    return crew

#def run_analysis(stock_symbol, model_name="ollama/llama3.2", provider="local"):
def run_analysis(stock_symbol, model_name, provider):

    #rew = create_crew(stock_symbol)
    #result = crew.kickoff()

    # Convert CrewOutput to a JSON-compatible dictionary
    """ if hasattr(result, "to_dict"):
        return result.to_dict()  # Preferred if CrewOutput provides this method
    else:
        # Fallback: Extract attributes manually
        return {
            "research_task": getattr(result, "research_task", "No research data available"),
            "sentiment_task": getattr(result, "sentiment_task", "No sentiment data available"),
            "analysis_task": getattr(result, "analysis_task", "No analysis data available"),
            "strategy_task": getattr(result, "strategy_task", "No strategy data available"),
        } """
    
    #crew = create_crew(stock_symbol, model_name="ollama/llama3.2", provider="local")
    crew = create_crew(stock_symbol, model_name, provider)
    print("Crew Created:", crew)
    result = crew.kickoff()


    analysis = {
        "technical_analysis": None,
        "chart_patterns": None,
        "sentiment_analysis": None,
        "risk_assessment": None,
        "competitor_analysis": None,
        "investment_strategy": None,
    }
    for idx, task_output in enumerate(result.tasks_output):
        # Map tasks to their respective outputs
        analysis[f"task_{idx}_description"] = task_output.description
        analysis[f"task_{idx}_raw"] = task_output.raw
        analysis[f"task_{idx}_summary"] = task_output.summary

    return analysis

    #print("Crew Kickoff Result:", result)
    #return result

# def run_analysis(stock_symbol, model_name, provider):
#     # Initialize the crew
#     crew = create_crew(stock_symbol, model_name, provider)
#     print("Crew Created:", crew)

#     #  Gather technical data from your tool
#     technical_data_result = yf_tech_analysis.run(stock_symbol)

#     # Kickoff passing real technical data
#     result = crew.kickoff(inputs={"technical_data": technical_data_result})

#     # Process result
#     analysis = {}
#     for idx, task_output in enumerate(result.tasks_output):
#         analysis[f"task_{idx}_description"] = task_output.description
#         analysis[f"task_{idx}_raw"] = task_output.raw
#         analysis[f"task_{idx}_summary"] = task_output.summary

#     return analysis


def analyze_indicator_with_crew(indicator, ticker, indicator_data, model_name, provider):


    llm = init_llm(model_name, provider)
    print("LLM initialized successfully.")
    #llm = LLM(
    #model='ollama/llama3.2',
    #base_url='http://localhost:11434',
    #temperature=0.7,
    #verbose=True
    #)   

    try:
        test_response = llm.invoke("Hello, can you respond?")
        print("LLM Test Response:", test_response)
    except Exception as e:
        print(f"LLM Initialization Error: {e}")
    # Debug LLM Initialization
    print("LLM initialized:", llm)

    indicator_analyst = Agent(
        role="Financial Indicator Analyst",
        goal="Provide detailed explanations and analysis for financial indicators.",
        backstory="You are an expert in financial analysis, specializing in technical indicators.",
        tools=[],
        memory=False,
        llm = llm,
        verbose=False
    )

    analyze_task = Task(
        description=(
            f"Analyze the financial indicators for the stock {ticker}.\n\n"
            f"Here are the calculated indicator values:\n{indicator_data}\n\n"
            f"Explain the implications of these values for the stock's performance and provide actionable insights."
        ),
        expected_output="A clear and concise explanation of the indicator for stock {ticker}.",
        agent=indicator_analyst
    )

    # indicator_analyst = Agent(
    # role="Technikai indikátor elemző",
    # goal="Részletes magyarázatok és elemzések készítése pénzügyi indikátorokról.",
    # backstory="Tapasztalt pénzügyi elemző vagy, aki a technikai indikátorokra specializálódott.",
    # tools=[],
    # llm = llm,
    # memory=False,
    # verbose=False
    # )

    # analyze_task = Task(
    #     description=(
    #         "Elemezd a(z) {indicator} indikátort a(z) {ticker} részvény esetében. "
    #         "Magyarázd el, hogy mit mér ez az indikátor, hogyan számítják ki, "
    #         "és mit jelezhet a részvény teljesítményére vonatkozóan." \
    #         "Válaszodat kizárólag magyar nyelven add meg."
    #     ),
    #     expected_output="Egy világos és tömör magyarázat a(z) {ticker} részvényhez kapcsolódó indikátorról.",
    #     agent=indicator_analyst
    # )

    crew = Crew(
        agents=[indicator_analyst],
        tasks=[analyze_task],
        process=Process.sequential
    )

    result = crew.kickoff(inputs={"indicator": indicator, "ticker": ticker})

    # Prepare the analysis dictionary
    analysis = {}
    for idx, task_output in enumerate(result.tasks_output):
        analysis[f"task_{idx}_description"] = task_output.description
        analysis[f"task_{idx}_raw"] = task_output.raw
        analysis[f"task_{idx}_summary"] = task_output.summary

    return analysis
