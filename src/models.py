import os
import json
import streamlit as st
from huggingface_hub import InferenceClient

class LLMEngine:
    def __init__(self, api_token=None):
        self.repo_id = "Qwen/Qwen2.5-7B-Instruct"
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")

        if not self.api_token:
            raise ValueError("No HuggingFace Token provided")

        self.client = InferenceClient(
            model=self.repo_id,
            token=self.api_token
        )

    def _get_system_prompt(self, prompt_type):
        if prompt_type == "scoring":
            return """
            You are an expert Hong Kong secondary school Chinese teacher. 
            Grade the following essay (0-100).
            You MUST return the result in valid JSON format ONLY. 
            Do not write any introduction or conclusion.
            Structure:
            {
                "holistic_score": <int>,
                "dimensions": {
                    "Content": <int>,
                    "Structure": <int>,
                    "Language": <int>,
                    "Creativity": <int>
                },
                "brief_comment": "<string>"
            }
            """
        elif prompt_type == "feedback":
            return """
            You are a writing coach. 
            Provide 3 specific sentence corrections and 3 suggestions.
            You MUST return the result in valid JSON format ONLY.
            Structure:
            {
                "corrections": [
                    {"quote": "<original>", "fix": "<correction>", "reason": "<explanation>"}
                ],
                "suggestions": ["<suggestion1>", "<suggestion2>", "<suggestion3>"]
            }
            """

    def analyze_essay(self, text, genre, prompt_type="scoring"):
        system_prompt = self._get_system_prompt(prompt_type)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Genre: {genre}\n\nEssay:\n{text}"}
        ]

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            raw_content = response.choices[0].message.content.strip()

            if raw_content.startswith("```"):
                raw_content = raw_content.split("```")[1]
                if raw_content.startswith("json"):
                    raw_content = raw_content[4:]

            return json.loads(raw_content)

        except Exception as e:
            return {"error": str(e)}
