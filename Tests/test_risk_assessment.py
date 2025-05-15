from tools.risk_assessment_tool import risk_assessment

""" # Test technical analysis tool
print("Technical Analysis Test:")
tech_data = sentiment_analysis("AAPL")
print(tech_data) """

result = risk_assessment.run("AAPL")
print(result)