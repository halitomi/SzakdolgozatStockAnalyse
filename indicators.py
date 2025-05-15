import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
import streamlit as st
import crewai
import matplotlib.pyplot as plt
from crew import analyze_indicator_with_crew
import time

# Fetch stock data
def fetch_stock_data(ticker, period):
    data = yf.download(ticker, period=period)

    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            data['Adj Close'] = data['Close']
        else:
            raise ValueError(f"'Close' and 'Adj Close' not found for {ticker}")
    return data

# Indicator Calculations
# def calculate_rsi(stock_data):
#     delta = stock_data['Adj Close'].diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     avg_gain = gain.rolling(window=14).mean()
#     avg_loss = loss.rolling(window=14).mean()
#     rs = avg_gain / avg_loss
#     stock_data['RSI'] = 100 - (100 / (1 + rs))
#     return stock_data
def calculate_rsi(stock_data, period=14):
    delta = stock_data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    stock_data['RSI'] = rsi
    return stock_data


def calculate_macd(stock_data):
    short_ema = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
    long_ema = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = short_ema - long_ema
    stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    return stock_data

def calculate_moving_averages(stock_data):
    if len(stock_data) < 200:
        st.warning("Insufficient data for 200-day Moving Average")
    stock_data['MA50'] = stock_data['Adj Close'].rolling(window=50, min_periods=1).mean()
    stock_data['MA200'] = stock_data['Adj Close'].rolling(window=200, min_periods=1).mean()
    return stock_data

def calculate_bollinger_bands(stock_data, window=20):
    if len(stock_data) < window:
        st.warning(f"Insufficient data for {window}-day Bollinger Bands")
    stock_data['SMA'] = stock_data['Adj Close'].rolling(window=window, min_periods=1).mean()
    stock_data['StdDev'] = stock_data['Adj Close'].rolling(window=window, min_periods=1).std()
    stock_data['Upper Band'] = stock_data['SMA'] + (2 * stock_data['StdDev'])
    stock_data['Lower Band'] = stock_data['SMA'] - (2 * stock_data['StdDev'])
    return stock_data

#def calculate_stochastic(stock_data):
    low_14 = stock_data['Low'].rolling(window=14).min()
    high_14 = stock_data['High'].rolling(window=14).max()
    stock_data['%K'] = 100 * ((stock_data['Adj Close'] - low_14) / (high_14 - low_14))
    stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()
    return stock_data

def calculate_stochastic(stock_data):
    low_14 = stock_data['Low'].rolling(window=14).min()
    high_14 = stock_data['High'].rolling(window=14).max()

    # Check and ensure single Series
    if isinstance(low_14, pd.DataFrame):
        low_14 = low_14.iloc[:, 0]
    if isinstance(high_14, pd.DataFrame):
        high_14 = high_14.iloc[:, 0]

    stock_data['%K'] = 100 * ((stock_data['Adj Close'] - low_14) / (high_14 - low_14))
    stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()

    return stock_data

# def calculate_parabolic_sar(stock_data):

#     try:
#         # Ensure required columns are present
#         required_columns = ['High', 'Low', 'Close']
#         missing_columns = [col for col in required_columns if col not in stock_data.columns]
#         if missing_columns:
#             raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

#         # Drop rows with missing values in High, Low, and Close
#         stock_data = stock_data.dropna(subset=required_columns)

#         # Calculate Parabolic SAR
#         indicator_psar = ta.trend.PSARIndicator(
#             high=stock_data['High'],
#             low=stock_data['Low'],
#             close=stock_data['Close'],
#             step=0.02,
#             max_step=0.2
#         )
#         stock_data['Parabolic SAR'] = indicator_psar.psar()

#     except Exception as e:
#         st.error(f"Error calculating Parabolic SAR: {e}")

#     return stock_data

def calculate_parabolic_sar(stock_data):
    try:
        df = stock_data.copy()
        
        # Convert to 1D arrays by flattening and ensuring proper Series creation
        high_series = pd.Series(df['High'].values.flatten(), index=df.index)
        low_series = pd.Series(df['Low'].values.flatten(), index=df.index)
        close_series = pd.Series(df['Close'].values.flatten(), index=df.index)
        
        # Verify the data is 1D
        if len(high_series.shape) != 1 or len(low_series.shape) != 1 or len(close_series.shape) != 1:
            raise ValueError("Failed to convert data to 1D series")
        
        # Calculate Parabolic SAR
        indicator_psar = ta.trend.PSARIndicator(
            high=high_series,
            low=low_series,
            close=close_series,
            step=0.02,
            max_step=0.2
        )
        
        # Calculate and store the PSAR values
        df['Parabolic SAR'] = indicator_psar.psar()
        
        # Handle any NaN values
        df['Parabolic SAR'] = df['Parabolic SAR'].fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating Parabolic SAR: {str(e)}")
        df['Parabolic SAR'] = None
        return df
    
    
# Plotting function with Plotly
def plot_indicator(stock_data, ticker, indicator):
    fig = go.Figure()

    if indicator == 'RSI':
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], name='RSI', line=dict(color='brown')))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title=f"{ticker} Relative Strength Index (RSI)")

    elif indicator == 'MACD':
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], name='MACD', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal Line'], name='Signal Line', line=dict(color='red')))
        fig.update_layout(title=f"{ticker} MACD and Signal Line")

    elif indicator == 'Moving Averages':
        #fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], name='Adjusted Close', line=dict(color='blue')))
        #fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA50'], name='MA50', line=dict(color='orange')))
        #fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA200'], name='MA200', line=dict(color='green')))
        #fig.update_layout(title=f"{ticker} Price with MA50 and MA200")

        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['Adj Close'], name='Adjusted Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['MA50'], name='MA50', line=dict(color='orange')))
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['MA200'], name='MA200', line=dict(color='green')))

        fig.update_layout(
            title=f"{ticker} Price with MA50 and MA200",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )
        
    elif indicator == 'Bollinger Bands':
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], name='Adj Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper Band'], name='Upper Band', line=dict(color='cyan')))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower Band'], name='Lower Band', line=dict(color='magenta')))
        fig.update_layout(title=f"{ticker} Bollinger Bands", xaxis_title="Date", yaxis_title="Price")

    elif indicator == 'Stochastic':
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%K'], name='%K', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%D'], name='%D', line=dict(color='orange')))
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title=f"{ticker} Stochastic Oscillator")

    # elif indicator == 'Parabolic SAR':
    #     plt.figure(figsize=(12, 6))

    #     plt.plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close', color='blue', linewidth=2)

    #     if 'Parabolic SAR' in stock_data.columns and stock_data['Parabolic SAR'].notna().any():
    #         plt.scatter(
    #             stock_data.index, stock_data['Parabolic SAR'],
    #             label='Parabolic SAR', color='purple', s=20, marker='o'
    #         )
    #         plt.title(f"{ticker} Parabolic SAR")
    #     else:
    #         plt.title(f"{ticker} Price Chart (Parabolic SAR Unavailable)")

    #     plt.xlabel("Date")
    #     plt.ylabel("Price")
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.tight_layout()

    #     plt.show()

        
    fig.update_layout(xaxis_title="Date", yaxis_title="Value", template="plotly_white")
    return fig

def plot_parabolic_sar(stock_data, ticker):
    
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close', color='blue', linewidth=2)

    if 'Parabolic SAR' in stock_data.columns and stock_data['Parabolic SAR'].notna().any():
        plt.scatter(
            stock_data.index,
            stock_data['Parabolic SAR'],
            label='Parabolic SAR',
            color='purple',
            s=20,
            marker='o'
        )
        plt.title(f"{ticker} Parabolic SAR", fontsize=14)
    else:
        plt.title(f"{ticker} Price Chart (Parabolic SAR Unavailable)", fontsize=14)

    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()

def plot_moving_averages(stock_data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close', color='blue', alpha=0.7)
    
    if 'MA50' in stock_data.columns:
        plt.plot(stock_data.index, stock_data['MA50'], label='MA50', color='orange', alpha=0.7)
    if 'MA200' in stock_data.columns:
        plt.plot(stock_data.index, stock_data['MA200'], label='MA200', color='green', alpha=0.7)
    
    plt.title(f"{ticker} Price with MA50 and MA200", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

def plot_bollinger_bands(stock_data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Adj Close'], label='Adjusted Close', color='blue', alpha=0.7)
    plt.plot(stock_data.index, stock_data['Upper Band'], label='Upper Band', color='cyan', alpha=0.7)
    plt.plot(stock_data.index, stock_data['Lower Band'], label='Lower Band', color='magenta', alpha=0.7)
    plt.fill_between(stock_data.index, stock_data['Upper Band'], stock_data['Lower Band'], color='gray', alpha=0.1)
    
    plt.title(f"{ticker} Bollinger Bands", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

# Indicator descriptions
def get_indicator_description(indicator):
    descriptions = {
        'RSI': """**Relative Strength Index (RSI)**: A momentum indicator measuring the magnitude of recent price changes. RSI values above 70 indicate overbought conditions, while values below 30 suggest oversold conditions.""",
        'MACD': """**Moving Average Convergence Divergence (MACD)**: A trend-following indicator showing the relationship between two moving averages of a security’s price. It is used to identify momentum and trend reversals.""",
        'Moving Averages': """**Moving Averages**: MA50 and MA200 represent the 50-day and 200-day simple moving averages, respectively. They help identify overall trend directions.""",
        'Bollinger Bands': """**Bollinger Bands**: A volatility indicator showing an upper and lower band around the moving average. Prices near the upper band may indicate overbought conditions, and prices near the lower band may indicate oversold conditions.""",
        'Stochastic': """**Stochastic Oscillator**: A momentum indicator comparing a security’s closing price to its price range over a given period. Values above 80 indicate overbought conditions, while values below 20 suggest oversold conditions.""",
        'Parabolic SAR': """**Parabolic SAR**: A trend-following indicator placing dots above or below price to signal potential reversals. Dots below the price suggest an uptrend, and dots above indicate a downtrend."""
    }
    return descriptions.get(indicator, "No description available for this indicator.")


def run_indicators():
    st.title("Indicator analysis")
    st.sidebar.header("Configuration")
    stock_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL")
    period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "5y", "max"])
    indicator = st.sidebar.selectbox(
        "Indicator", 
        ['RSI', 'MACD', 'Moving Averages', 'Bollinger Bands', 'Stochastic', 'Parabolic SAR']
    )
    
    model_name = st.session_state.selected_model
    provider = st.session_state.provider

    if st.sidebar.button("Analyze"):
        with st.spinner("Fetching data, calculating indicators and generating description..."):
            stock_data = fetch_stock_data(stock_symbol, period)

            if not stock_data.empty:
                # Calculate selected indicator
                if indicator == 'RSI':
                    stock_data = calculate_rsi(stock_data)
                    fig = plot_indicator(stock_data, stock_symbol, indicator)
                    st.plotly_chart(fig)
                elif indicator == 'MACD':
                    stock_data = calculate_macd(stock_data)
                    fig = plot_indicator(stock_data, stock_symbol, indicator)
                    st.plotly_chart(fig)
                elif indicator == 'Moving Averages':
                    stock_data = calculate_moving_averages(stock_data)
                    plot_moving_averages(stock_data, stock_symbol)
                elif indicator == 'Bollinger Bands':
                    stock_data = calculate_bollinger_bands(stock_data)
                    plot_bollinger_bands(stock_data, stock_symbol)
                elif indicator == 'Stochastic':
                    stock_data = calculate_stochastic(stock_data)
                    fig = plot_indicator(stock_data, stock_symbol, indicator)
                    st.plotly_chart(fig)
                elif indicator == 'Parabolic SAR':
                    stock_data = calculate_parabolic_sar(stock_data)
                    # fig = plot_indicator(stock_data, stock_symbol, indicator)
                    fig = plot_parabolic_sar(stock_data, stock_symbol)
                    # st.plotly_chart(fig)

                # Display indicator description
                #crew_analysis = analyze_indicator_with_crew(indicator, ticker, stock_data)
                crew_analysis = analyze_indicator_with_crew(indicator, stock_symbol, stock_data, model_name, provider)
                st.write(crew_analysis.get("task_0_raw", "No indicator analysis available"))
            else:
                st.error("No data found for the selected ticker and period.")
