import sys
import os

# Ensure the project root is in the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.llm import call_llm

def main():
    print("="*60)
    print("MANUAL MODEL TEST")
    print("="*60)
    print("Attempting to load Qwen 3B model...")
    print("(If it's the first time, this will take 2-5 mins to download)")
    print("="*60 + "\n")
    
    # We use a unique prompt to bypass any old cached mock responses
    system_prompt = "You are a clinical AI assistant. Answer concisely."
    user_prompt = "List 3 common symptoms of GERD and how it is treated."
    
    print("Sending prompt to the local model...")
    
    response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        mock=False,       # Force real model
        max_tokens=150
    )
    
    print("\n" + "="*60)
    print("RAW MODEL RESPONSE:")
    print("="*60)
    print(response)
    print("="*60)

if __name__ == "__main__":
    main()