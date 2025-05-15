from crewai import LLM
import os
from dotenv import load_dotenv

load_dotenv()

def init_llm(model_name, provider):
    if provider == "openrouter":
        return LLM(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.7,
            verbose=True,
            stream = False
        )
    elif provider == "local":
        return LLM(
            model="ollama/llama3.2", 
            base_url="http://localhost:11434",
            temperature=0.7,
            provider="ollama",
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
