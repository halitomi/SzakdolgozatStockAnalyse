from transformers import BertTokenizer, BertForSequenceClassification
import torch
import praw
import requests
from datetime import datetime, timedelta
from crewai_tools import tool
import json
import os
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("MARKETAUX_API_KEY")

news_model_path = 'saved_model_news'
reddit_model_path = 'fine_tuned_reddit_model'

news_model = BertForSequenceClassification.from_pretrained(news_model_path)
news_tokenizer = BertTokenizer.from_pretrained(news_model_path)
reddit_model = BertForSequenceClassification.from_pretrained(reddit_model_path)
reddit_tokenizer = BertTokenizer.from_pretrained(reddit_model_path)

news_model.eval()
reddit_model.eval()


# Unified sentiment analysis tool
@tool
def sentiment_analysis(ticker: str):
    """
    Perform sentiment analysis using news and Reddit data.
    Args:
        ticker (str): Stock ticker.
    Returns:
        dict: Sentiment analysis results.
    """
    print(f"[TOOL CALL] sentiment_analysis used with ticker={ticker}")
    news_api_key=api_key
    subreddits = ['wallstreetbets', 'stocks', 'investing', 'pennystocks']
    limit=10
    days=7
    # 1. Fetch News and Predict Sentiment
    #news_sentiments = analyze_news_sentiment(ticker, news_api_key)
    try:
        news_sentiments = analyze_news_sentiment(ticker, news_api_key)
        news_avg_sentiment = sum(news_sentiments) / len(news_sentiments) if news_sentiments else None
    except Exception as e:
        print(f"[NEWS] Failed to fetch or process news: {e}")
        news_avg_sentiment = None

    # 2. Fetch Reddit Posts and Predict Sentiment
    #reddit_sentiments = analyze_reddit_sentiment(ticker, subreddits, limit, days)
    try:
        reddit_sentiments = analyze_reddit_sentiment(ticker, subreddits, limit, days)
        reddit_avg_sentiment = sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else None
    except Exception as e:
        print(f"[REDDIT] Failed to fetch or process Reddit data: {e}")
        reddit_avg_sentiment = None

    # 3. Aggregate Results
    #news_avg_sentiment = sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0
    #reddit_avg_sentiment = sum(reddit_sentiments) / len(reddit_sentiments) if reddit_sentiments else 0

    #overall_sentiment = (news_avg_sentiment + reddit_avg_sentiment) / 2
    valid_sources = [s for s in [news_avg_sentiment, reddit_avg_sentiment] if s is not None]
    overall_sentiment = sum(valid_sources) / len(valid_sources) if valid_sources else None

    return {
        "ticker": ticker,
        "news_sentiment": news_avg_sentiment,
        "reddit_sentiment": reddit_avg_sentiment,
        "overall_sentiment": overall_sentiment
    }

    """ return {
        f"ticker: {ticker}\n"
        f"news_sentiment: {news_avg_sentiment}\n"
        f"reddit_sentiment: {reddit_avg_sentiment}\n"
        f"overall_sentiment: {overall_sentiment}\n"
    } """


# News Sentiment Analysis
def analyze_news_sentiment(ticker, api_key):
    published_after = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    articles = fetch_news(api_key, ticker, published_after)
    inputs = preprocess_articles(articles, news_tokenizer)
    return predict_sentiment_news(inputs, news_model)


# Reddit Sentiment Analysis
def analyze_reddit_sentiment(ticker, subreddit, limit, days):
    oldest_date = datetime.now() - timedelta(days=days)
    posts = fetch_reddit_posts(subreddit, ticker, limit, oldest_date)
    sentiments = []
    for post in posts:
        text = post['title'] + ' ' + post['selftext']
        sentiment, _ = predict_sentiment(text, reddit_model, reddit_tokenizer)
        sentiments.append(sentiment)
    return sentiments


# Helper Functions (News)
def fetch_news(api_key, query, published_after, language='en', num_articles=10):
    url = 'https://api.marketaux.com/v1/news/all'
    articles = []
    page = 1
    limit = 3
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
            if not data:
                break
            articles.extend(data)
            page += 1
        else:
            break
    return articles[:num_articles]


def preprocess_articles(articles, tokenizer, max_len=128):
    inputs = []
    for article in articles:
        text = article.get('description', '')
        if not text:
            continue
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
    if len(inputs) == 0:
        raise ValueError("No valid inputs were processed for tokenization.")
    return inputs


def predict_sentiment_news(inputs, model):
    sentiments = []
    with torch.no_grad():
        for input_data in inputs:
            input_ids = input_data['input_ids'].unsqueeze(0)
            attention_mask = input_data['attention_mask'].unsqueeze(0)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            sentiment = torch.argmax(logits, dim=1).item()
            sentiments.append(sentiment)
    return sentiments


# Helper Functions (Reddit)
def fetch_reddit_posts(subreddits, ticker, limit, oldest_date):
    """
    Fetch Reddit posts from multiple subreddits for a specific stock ticker.

    Args:
        subreddits (list): List of subreddit names to search.
        ticker (str): Stock ticker to search for in Reddit posts.
        limit (int): Maximum number of posts to fetch per subreddit.
        oldest_date (datetime): The oldest date to include posts from.

    Returns:
        list: A list of dictionaries containing post details from all subreddits.
    """
    reddit = praw.Reddit(client_id='BwmAQNDRFInXeVqw98lYew',
                         client_secret='C3rFb_Kixdt9Y51HuI_-Z9o9So2JtQ',
                         user_agent='Stockwebscrape')
    
    posts = []
    for subreddit in subreddits:
        for submission in reddit.subreddit(subreddit).search(ticker, limit=limit, sort='new'):
            post_date = datetime.fromtimestamp(submission.created_utc)
            if post_date >= oldest_date:
                posts.append({
                    'subreddit': subreddit,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'url': submission.url,
                    'date': post_date
                })
    return posts


def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    sentiment = torch.argmax(logits, dim=1).item()
    return sentiment, torch.softmax(logits, dim=1).cpu().numpy()
