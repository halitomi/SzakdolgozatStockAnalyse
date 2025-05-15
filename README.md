# SzakdolgozatStockAnalyse


Content-Based Stock Forecasting System

A modular stock analysis platform that combines BERT-based sentiment analysis, technical indicators, and LLM-driven reasoning agents to support investment decision-making. Developed as part of a bachelor's thesis at Óbuda University.

---

Overview

The system performs stock forecasting by:
- Analyzing market sentiment from financial **news** and **Reddit** posts using fine-tuned BERT models
- Calculating technical indicators from **Yahoo Finance**
- Interpreting findings through a **CrewAI agent system** powered by local (Ollama) or API-based LLMs (via OpenRouter)
- Displaying results interactively through a **Streamlit** user interface

---

Features

-  BERT-based sentiment models fine-tuned on domain-specific Reddit and news datasets
-  Technical indicator calculations: RSI, MACD, MA50/MA200, Bollinger Bands, SAR, Stochastic
-  CrewAI multi-agent system for layered financial interpretation and investment strategy generation
-  Optional LLM model selection between local (LLaMA 3.2 via Ollama) and external (GPT-4, Claude, etc.)
-  Fully modular Streamlit dashboard

---

How to Run the Application

1. Clone the Repository
bash
git clone https://github.com/your-username/SzakdolgozatStockAnalyse.git
cd SzakdolgozatStockAnalyse

2. Set Up a Virtual Environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

3. Install Requirements
pip install -r requirements.txt

4. API Keys
.env.example
MARKETAUX_API_KEY=your_marketaux_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=Stockwebscrape
OPENROUTER_API_KEY=your_openrouter_key

5. Launch the App
streamlit run app.py

project/
├── app.py                          # Streamlit frontend
├── main_llm.py                     # LLM/CrewAI-based analysis logic
├── crew.py                         # CrewAI agent/task setup
├── llm_config.py                   # Model and provider config
├── indicators.py                   # RSI, MACD, MA50/200, etc.
├── bert_news.py                    # News sentiment prediction
├── bert_reddit.py                  # Reddit sentiment prediction
├── bert_sentiment_news.py          # (Optional separate loader)
├── tools/
│   ├── sentiment_analysis_tool.py
│   ├── yf_fundamental_analysis_tool.py
│   ├── yf_tech_analysis_tool.py
│   ├── risk_assessment_tool.py
│   └── competitor_analysis_tool.py
├── saved_model_news/               # Fine-tuned BERT model for news
├── fine_tuned_reddit_model/        # Fine-tuned BERT model for Reddit
├── .env.example                    # Template for environment setup
├── .gitignore                      # Files to exclude from Git
├── requirements.txt                # All dependencies
└── README.md

Notes

Ollama must be running locally to use the LLaMA 3.2 model.

OpenRouter allows remote access to OpenAI, Claude, and Gemini models if configured.

This project requires two fine-tuned BERT models for sentiment analysis.

Download them from Google Drive:
https://drive.google.com/drive/folders/1A0l0vaYacPwBi7ICQ9h6fRkHScNmayJF?usp=drive_link
After downloading:

1. Extract each folder
2. Place them into the root project directory so that your folder looks like this:

project/
├── saved_model_news/
│ ├── config.json
│ ├── pytorch_model.bin
│ └── ...
├── fine_tuned_reddit_model/
│ ├── config.json
│ ├── pytorch_model.bin
│ └── ...
├── app.py
├── ...

Author
Tamas Halasz
Bachelor’s thesis – Engineering Informatics BSc
Óbuda University – 2025
