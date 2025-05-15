import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import json
from crew import run_analysis
import time

#def main_llm(model=None, provider=None):
def main_llm(model, provider):
    st.title("Agent Analysis")

    # User input
    stock_symbol = st.text_input("Enter a stock symbol (For example, AAPL):", "AAPL")
    
    if st.button("Analyse"):
        with st.spinner("Analysing..."):
                try:
                    ticker_obj = yf.Ticker(stock_symbol)
                    data = ticker_obj.history(period="1y")

                    if data.empty:
                        st.error(f"The ticker '{stock_symbol}' is invalid or has no historical data.")
                        return

                except Exception as e:
                    st.error(f"Failed to fetch data for '{stock_symbol}': {e}")
                    return

                result = run_analysis(stock_symbol, model_name=model, provider=provider)


        

        st.success("Analysis ready!")

        
        """
        result_json = json.dumps(result, default=lambda o: o.__dict__, indent=4)
        analysis = json.loads(result_json)
        # --- Technikai elemzés ---
        st.subheader("Technikai elemzés")
        tech_placeholder = st.empty()
        tech_partial = ""
        tech_text = analysis.get("task_0_raw", "Nem sikerült a technikai elemzés")

        for token in tech_text:
            tech_partial += token
            tech_placeholder.markdown(tech_partial)
            time.sleep(0.01)

        # --- Szentimentális elemzés ---
        st.subheader("Szentimentális elemzés")
        sentiment_placeholder = st.empty()
        sentiment_partial = ""
        sentiment_text = analysis.get("task_1_raw", "Nem sikerült a szentimentális elemzés")

        for token in sentiment_text:
            sentiment_partial += token
            sentiment_placeholder.markdown(sentiment_partial)
            time.sleep(0.01)

        # --- Kockázati elemzés ---
        st.subheader("Kockázati elemzés")
        risk_placeholder = st.empty()
        risk_partial = ""
        risk_text = analysis.get("task_2_raw", "Nem sikerült a kockázati elemzés")

        for token in risk_text:
            risk_partial += token
            risk_placeholder.markdown(risk_partial)
            time.sleep(0.01)

        # --- Befektetési javaslat ---
        st.subheader("Befektetési javaslat")
        strategy_placeholder = st.empty()
        strategy_partial = ""
        strategy_text = analysis.get("task_3_raw", "Nem sikerült a befektetési elemzés")

        for token in strategy_text:
            strategy_partial += token
            strategy_placeholder.markdown(strategy_partial)
            time.sleep(0.01)"""
            
        result_json = json.dumps(result, default=lambda o: o.__dict__, indent=4)
        analysis = json.loads(result_json)
        
        
        # Display analysis result
        st.header("Analysis report")

        st.subheader("Technical analysis")
        st.write(analysis.get("task_0_raw", "Nem sikerült a technikai elemzés"))

        st.subheader("Sentiment analysis")
        st.write(analysis.get("task_1_raw", "Nem sikerült a szentimentális elemzés"))

        st.subheader("Risk analysis")
        st.write(analysis.get("task_2_raw", "Nem sikerült a kockázati elemzés"))

        st.subheader("Strategy suggestion")
        st.write(analysis.get("task_3_raw", "Nem sikerült a befektetési elemzés"))
        

        stock = yf.Ticker(stock_symbol)
        hist = stock.history(period="1y")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price'
        ))

        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', yaxis='y2'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(window=50).mean(), name='50-day MA'))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(window=200).mean(), name='200-day MA'))

        fig.update_layout(
            title=f"{stock_symbol} Stock analyse",
            yaxis_title='Price',
            yaxis2=dict(title='Volume', overlaying='y', side='right'),
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Main statistics")
        info = stock.info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
            st.metric("P/E Ratio", round(info.get('trailingPE', 0), 2))
        with col2:
            st.metric("52 Week High", f"${info.get('fiftyTwoWeekHigh', 0):,.2f}")
            st.metric("52 Week Low", f"${info.get('fiftyTwoWeekLow', 0):,.2f}")
        with col3:
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0):.2%}")
            st.metric("Beta", round(info.get('beta', 0), 2))

