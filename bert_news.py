import streamlit as st
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit_shadcn_ui as ui
import os
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("MARKETAUX_API_KEY")



# Load your trained model and tokenizer
#model_path = 'D:/saved_model_news'
model_path = 'saved_model_news'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Function to fetch news articles
def fetch_news(api_key, query, published_after, language='en', num_articles=10):
    url = f'https://api.marketaux.com/v1/news/all'
    articles = []
    page = 1
    limit = 3  # MarketAux fetches 3 articles per page

    while len(articles) < num_articles:
        params = {
            'api_token': api_key,
            'search': query,
            'language': language,
            'page': page,
            'limit': limit,
            'sort_by': 'published_desc',
            'published_after': published_after,  
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json().get('data', [])
            if not data:  # Stop if no more articles are available
                break
            articles.extend(data)
            page += 1  # Move to the next page
        else:
            st.error(f"Error fetching news: {response.status_code}")
            break

    return articles[:num_articles]  # Return the exact number of requested articles

# Function to preprocess articles for the model
def preprocess_articles(articles, max_len=128):
    inputs = []
    for article in articles:
        text = article.get('description', '')
        if text:
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            inputs.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            })
    return inputs

# Function to predict sentiment
def predict_sentiment(inputs):
    sentiments = []
    with torch.no_grad():
        for input_data in inputs:
            input_ids = input_data['input_ids'].unsqueeze(0)  # Add batch dimension
            attention_mask = input_data['attention_mask'].unsqueeze(0)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            sentiments.append(prediction)
    return sentiments


def format_date(date_string):
    try:
        # Convert the ISO 8601 datetime string to a datetime object
        date = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S.%fZ')
        return date.strftime('%Y-%m-%d')  # Format only the date part
    except Exception:
        return "No Date"
    
import matplotlib.pyplot as plt

# Prepare and display a pie chart
def display_sentiment_pie_chart(sentiments):
    # Count positive and negative sentiments
    sentiment_labels = ['Negative', 'Positive']
    sentiment_counts = [sentiments.count(0), sentiments.count(1)]
    
    # Create a Plotly Pie Chart
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_labels,
        values=sentiment_counts,
        hole=0.4,  # Makes it a donut chart
        textinfo='percent+label',  # Show percentage and label
        marker=dict(colors=['#FF6961', '#77DD77']),  # Custom colors
    )])
    fig.update_layout(
        title='',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


def run_news_bert_model():
    # Streamlit UI
    st.title("Stock News Sentiment Analysis")
    st.markdown("Analyze financial news using a fine-tuned BERT model")

    query = st.text_input("Enter stock symbol (e.g., 'AAPL'):")
    num_articles = st.number_input("Number of articles (3–50):", min_value=3, max_value=50, value=10)
    published_after = st.date_input("Only show news from this date onward:", value=datetime.now())

    if st.button("Analyze"):
        if not query:
            st.error("Please enter a stock symbol.")
        else:
            st.info("Fetching news...")
            formatted_date = published_after.strftime('%Y-%m-%d')  # Format the date 
            articles = fetch_news(api_key, query, published_after=formatted_date, num_articles=num_articles)

            if articles:
                st.success(f" {len(articles)} news articles found. Analyzing sentiment...")
                inputs = preprocess_articles(articles)
                sentiments = predict_sentiment(inputs)

                # Display all articles with additional information
                for article, sentiment in zip(articles, sentiments):
                    st.subheader(article.get('title', 'No Title'))
                    st.write(article.get('description', 'No Description'))
                    st.write(f"**Date:** {format_date(article.get('published_at', 'No Date'))}")
                    st.write(f"**Link:** [Read more]({article.get('url', '#')})")
                    st.write(f"**Sentiment:** {'Positive' if sentiment == 1 else 'Negative'}")
                    st.write("---")
            
                st.subheader("Sentiment Distribution")
                pie_chart = display_sentiment_pie_chart(sentiments)
                st.plotly_chart(pie_chart, use_container_width=True)        
            else:
                st.warning("No news found for the given symbol.")