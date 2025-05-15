import streamlit as st

st.set_page_config(layout="wide")
st.title("Stock analyser")

# Load your external modules or functions here
from bert_news import run_news_bert_model
from bert_reddit import run_reddit_bert_model
from indicators import run_indicators
from main_llm import main_llm



def setup_sidebar():
    st.sidebar.title("Navigation")


    # --- App Mode Selector ---
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "ollama/llama3.2"
    if "provider" not in st.session_state:
        st.session_state.provider = "local"

    # --- Model Selector ---
    model_map = {
        "LLaMA 3 (local)": "ollama/llama3.2",
        "GPT-4 (OpenAI)": "openai/gpt-4"
    }
    selected_label = st.sidebar.selectbox("💡 Choose an LLM model:", list(model_map.keys()))
    selected_model = model_map[selected_label]

    st.session_state.selected_model = selected_model
    st.session_state.provider = "local" if selected_model.startswith("ollama/") else "openrouter"

    if "app_mode" not in st.session_state:
        st.session_state.app_mode = None  # Initialize state if not already done

    if st.sidebar.button("Agent analysis", key="main_llm"):
        st.session_state.app_mode = "Agent analysis"
    if st.sidebar.button("Reddit BERT Model Sentiment analysis", key="reddit_bert"):
        st.session_state.app_mode = "Reddit BERT Model Sentiment analysis"
    if st.sidebar.button("News BERT Model Sentiment analysis", key="news_bert"):
        st.session_state.app_mode = "News BERT Model Sentiment analysis"
    if st.sidebar.button("Stock indicators", key="stock_indicators"):
        st.session_state.app_mode = "Stock indicators"
    st.sidebar.write("Selected Module: ", st.session_state.app_mode)  # Display the selected module for debugging
    return st.session_state.app_mode


def main():

    # Display the selected module
    selected_module = setup_sidebar()

    if selected_module == "Agent analysis":
        main_llm(model=st.session_state.selected_model,provider=st.session_state.provider)
    elif selected_module == "Reddit BERT Model Sentiment analysis":
        run_reddit_bert_model()
    elif selected_module == "News BERT Model Sentiment analysis":
        run_news_bert_model()
    elif selected_module == "Stock indicators":
        run_indicators()
    else:
        st.write("Choose a module")

if __name__ == "__main__":
    main()