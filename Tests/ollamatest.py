from langchain_ollama import OllamaLLM


try:
    llm = OllamaLLM(model="tinyllama", base_url="http://localhost:11434")
    response = llm("What is the capital of France?")
    print(response)
except Exception as e:
    print(f"Error: {e}")


import ollama

def ollama_llm_call(messages):
    response = ollama.chat(model='tinyllama', messages=messages)
    return response['message']['content']