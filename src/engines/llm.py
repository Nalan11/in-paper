import time
import json
from openai import OpenAI
from src.utils.validation import attempt_json_recovery, validate_extraction

class LLMEngine:
    def __init__(self, server_url="http://localhost:8001/v1", model_name="Qwen/Qwen3-4B-AWQ"):
        self.client = OpenAI(base_url=server_url, api_key="EMPTY")
        self.model_name = model_name

    def extract(self, text, system_prompt):
        start_time = time.time()
        
        messages_payload = [
            {"role": "system", "content": system_prompt + "\nBe as concise as possible to avoid truncation."},
            {"role": "user", "content": f"Text to process:\n{text}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_payload,
            temperature=0.1,
            # frequency_penalty REMOVED: It actively prevents the model from outputting JSON arrays because it punishes repeating keys like "description".
            response_format={"type": "json_object"}
        )
        
        raw_content = response.choices[0].message.content
        duration = time.time() - start_time
        
        try:
            result_json = json.loads(raw_content)
        except json.JSONDecodeError:
            result_json = attempt_json_recovery(raw_content)
            result_json["requires_human_review"] = True
            if "validation_errors" not in result_json:
                result_json["validation_errors"] = []
            result_json["validation_errors"].append("LLM output was truncated/incomplete")

        # Validation Gate
        result_json = validate_extraction(result_json)
        
        # Include diagnostics in the return tuple
        diagnostics = {
            "prompt_messages": messages_payload,
            "raw_output_string": raw_content
        }
        
        return result_json, duration, diagnostics
