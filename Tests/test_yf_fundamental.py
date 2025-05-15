""" from tools.yf_tech_analysis_tool import yf_tech_analysis

# Test technical analysis tool
print("Technical Analysis Test:")
tech_data = yf_tech_analysis.func("AAPL")
print(tech_data) """

from tools.yf_fundamental_analysis_tool import yf_fundamental_analysis
from Multi_Agent.Tests.sentiment_analysis_tool3 import sentiment_analysis

""" # Test technical analysis tool
print("Technical Analysis Test:")
tech_data = sentiment_analysis("AAPL")
print(tech_data) """

result = yf_fundamental_analysis.run("AAPL")
print(result)