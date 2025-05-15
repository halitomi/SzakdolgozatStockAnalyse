from tools.yf_tech_analysis_tool import yf_tech_analysis

""" # Test technical analysis tool
print("Technical Analysis Test:")
tech_data = sentiment_analysis("AAPL")
print(tech_data) """

result = yf_tech_analysis.run("AAPL",)
print(result)