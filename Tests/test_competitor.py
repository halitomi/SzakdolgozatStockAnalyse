from tools.competitor_analysis_tool import competitor_analysis

""" # Test technical analysis tool
print("Technical Analysis Test:")
tech_data = sentiment_analysis("AAPL")
print(tech_data) """

result = competitor_analysis.run("AApl")
print(result)