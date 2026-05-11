import os
import requests
import logging

logger = logging.getLogger(__name__)

class AnomalyReporter:
    def __init__(self):
        # I use Mistral-7B-Instruct-v0.2 on HF as primary to avoid local GPU bottlenecks.
        self.hf_api_key = os.getenv("HF_API_KEY", "")
        self.hf_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        # I configure a local fallback for offline robustness.
        self.ollama_url = "http://localhost:11434/api/generate"

    def generate_report(self, analytics_state: dict) -> str:
        prompt = self._build_prompt(analytics_state)
        
        # Primary: HuggingFace API
        if self.hf_api_key:
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.3}}
            try:
                response = requests.post(self.hf_url, headers=headers, json=payload, timeout=5.0)
                if response.status_code == 200:
                    result = response.json()
                    # Parse Mistral output
                    if isinstance(result, list) and 'generated_text' in result[0]:
                        text = result[0]['generated_text'].replace(prompt, '').strip()
                        if text:
                            return text
            except Exception as e:
                logger.warning(f"HF API failed: {e}. Falling back to Ollama.")

        # Fallback: Ollama
        try:
            payload = {"model": "llama3.2", "prompt": prompt, "stream": False}
            response = requests.post(self.ollama_url, json=payload, timeout=5.0)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"Ollama fallback failed: {e}")

        return "Report unavailable. Could not connect to AI services."

    def _build_prompt(self, analytics_state: dict) -> str:
        return f"""[INST] You are an expert retail store manager assistant. Analyze the following live analytics data from my Smart Retail Platform and generate a brief, professional 2-sentence summary identifying any anomalies or areas needing attention.

Analytics State:
{analytics_state}

Summary Report: [/INST]"""
