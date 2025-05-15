import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import praw
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import os


@st.cache_resource
def load_model():
    #model = BertForSequenceClassification.from_pretrained('D:\\Models\\fine_tuned_reddit_model')
    model = BertForSequenceClassification.from_pretrained('fine_tuned_reddit_model')
    #tokenizer = BertTokenizer.from_pretrained('D:\\Models\\fine_tuned_reddit_model')
    tokenizer = BertTokenizer.from_pretrained('fine_tuned_reddit_model')
    return model, tokenizer

model, tokenizer = load_model()

# Function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
    inputs = {key: value.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(probs, dim=1).item()
    return sentiment

# Function to fetch Reddit posts
def fetch_reddit_posts(subreddit, ticker, limit, oldest_date):
    reddit = praw.Reddit(
                        client_id=os.getenv('REDDIT_CLIENT_ID'),
                        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                        user_agent='Stockwebscrape')
    posts = []
    for submission in reddit.subreddit(subreddit).search(ticker, limit=limit, sort='new'):
        post_date = datetime.fromtimestamp(submission.created_utc)
        if post_date >= oldest_date:
            posts.append({
                'title': submission.title,
                'selftext': submission.selftext,
                'url': submission.url,
                'date': post_date
            })
    return posts

# Prepare and display a pie chart
def display_sentiment_pie_chart(sentiments):
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    sentiment_counts = [sentiments.count(0), sentiments.count(1), sentiments.count(2)]
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_labels,
        values=sentiment_counts,
        hole=0.4,  
        textinfo='percent+label', 
        marker=dict(colors=['#FF6961', '#FFD700', '#77DD77']) 
    )])
    
    # Update layout
    fig.update_layout(
        title='Sentiment Distribution',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def run_reddit_bert_model():
    # Streamlit UI
    st.title("Reddit Sentiment Analysis")
    ticker = st.text_input("Enter a stock symbol:")
    post_limit = st.number_input("Number of posts to analyze:", min_value=1, max_value=100, value=10, step=1)
    oldest_date = st.date_input("Start date:", value=datetime.today() - timedelta(days=7))

    if st.button("Analyze"):
        if ticker:
            st.write(f"Analyzing  {post_limit} posts about {ticker} since {oldest_date}")
            oldest_date = datetime.combine(oldest_date, datetime.min.time())
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'pennystocks']
            all_posts = []
            sentiments = []

            for subreddit in subreddits:
                if len(all_posts) >= post_limit: 
                    break
                remaining_limit = post_limit - len(all_posts) 
                posts = fetch_reddit_posts(subreddit, ticker, limit=remaining_limit, oldest_date=oldest_date)
                all_posts.extend(posts)

            for post in all_posts:
                text = f"{post['title']} {post['selftext']}"
                sentiment = predict_sentiment(text, model, tokenizer)
                sentiments.append(sentiment)
                st.subheader(post['title'])
                st.write(post['selftext'])
                st.write(f"**Date:** {post['date'].strftime('%Y-%m-%d')}")
                st.write(f"**Link:** [Read more]({post['url']})")
                st.write(f"**Sentiment:** {'Positive' if sentiment == 2 else 'Neutral' if sentiment == 1 else 'Negative'}")
                st.write("---")

            if sentiments:
                pie_chart = display_sentiment_pie_chart(sentiments)
                st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.warning("Please enter a stock symbol")
