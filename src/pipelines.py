from src.models import LLMEngine

class ScoringPipeline:
    def __init__(self, token):
        self.engine = LLMEngine(token)

    def run(self, text, genre):
        if not text.strip():
            return {"error": "Empty text"}
        return self.engine.analyze_essay(text, genre, prompt_type="scoring")

class FeedbackPipeline:
    def __init__(self, token):
        self.engine = LLMEngine(token)

    def run(self, text, genre, scores):
        if not text.strip() or "error" in scores:
            return {}

        context = f"Score: {scores.get('holistic_score')}/100. Essay:\n{text}"
        return self.engine.analyze_essay(context, genre, prompt_type="feedback")
