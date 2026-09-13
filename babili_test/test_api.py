import os
import sys
from pathlib import Path

# Add project root to sys.path so src imports work cleanly
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
load_dotenv(dotenv_path=root_dir / ".env", override=True)

from src.common.llm_client import get_llm_provider, last_call_metadata

def main():
    print("=== Testing 9Router API Connection ===")
    backend = os.getenv("LLM_BACKEND", "api")
    model = os.getenv("LLM_MODEL", "Ner-gm3.7m")
    base_url = os.getenv("LLM_API_BASE_URL", "http://localhost:20128/v1")
    
    print(f"Backend  : {backend}")
    print(f"Model    : {model}")
    print(f"Base URL : {base_url}")
    print("-" * 40)
    
    try:
        provider = get_llm_provider(backend=backend, model_name=model)
        prompt = "Hello! Please confirm you can receive and process this test message."
        print(f"Sending prompt: {prompt}\n")
        
        response = provider.generate(prompt=prompt, temperature=0.0, max_tokens=100)
        
        print("=== Response Received ===")
        print(response)
        print("-" * 40)
        print(f"Metadata: Latency={last_call_metadata.get('latency_seconds', 0.0):.2f}s, "
              f"Prompt Tokens={last_call_metadata.get('prompt_tokens')}, "
              f"Completion Tokens={last_call_metadata.get('completion_tokens')}")
        print("\nTest passed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Failed to send request: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
