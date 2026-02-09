# backend/tools/ollama_client.py
import os
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Optional

class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, timeout: int = 120):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate(self, model: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
            """
            Post to /api/generate. Returns plain text string.
            """
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model, 
                "prompt": prompt, 
                "stream": False,  # Important: disable streaming to get a single JSON response
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")
            except Exception as e:
                raise RuntimeError(f"Ollama generate failed: {e}")
