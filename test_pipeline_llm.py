import sys
import json
from src.engines.llm import LLMEngine

def test_llm(text_input):
    print(f"--- Testing LLM Component (Qwen3-4B-AWQ) ---")
    
    try:
        engine = LLMEngine(server_url="http://localhost:8001/v1")
    except Exception as e:
        print(f"Failed to connect to LLM server: {e}")
        print("Ensure 'start_servers.sh' is running or you have manually started vLLM on port 8001.")
        return
        
    print(f"Sending prompt to LLM...")
    try:
        result_json, duration, diagnostics = engine.extract(text_input)
        
        print(f"\n--- 1. Full Prompt Sent to API ---")
        print(json.dumps(diagnostics.get("prompt_messages", []), indent=2))
        
        print(f"\n--- 2. Raw LLM String Output (Duration: {duration:.2f}s) ---")
        print(diagnostics.get("raw_output_string", ""))
        
        print(f"\n--- 3. Parsed JSON Result ---")
        print(json.dumps(result_json, indent=2))
        
    except Exception as e:
        print(f"Error during LLM extraction: {e}")

if __name__ == "__main__":
    test_text = """
    Photography
    Invoice
    123 Street Name
    Denver, CO 80205
    P: 555-555-5555
    email@samplebusiness.com
    INVOICE
    Invoice #: 12074
    Invoice date: 7/19/24
    Job Details: Photo Shoot
    Bill to: Customer Name
    Address: 123 Street Name Denver, CO 80205
    Phone: 555-555-5555
    | Description | Qty | Unit price | Discount | Price |
    | --- | --- | --- | --- | --- |
    | Wedding Ceremony Photos | 1 | $1,500.00 |  | $1,500.00 |
    | Bride & Groom Portraits | 1 | $500.00 |  | $500.00 |
    | Engagement Photoshoot | 1 | $300.00 |  | $300.00 |
    |  |  |  |  | $0.00 |
    |  |  |  |  | $0.00 |
    |  |  |  |  | $0.00 |
    |  |  |  |  | $0.00 |
    |  |  |  |  | $0.00 |
    |  |  |  |  | Invoice Subtotal |
    |  |  |  | Tax Rate | $2,300.00 |
    |  |  |  | Sales Tax | $6.00% |
    |  |  |  | Deposit Received | $138.00 |
    |  |  |  | TOTAL | $2,388.00 |
    Please make all checks payable to Wedding Photography. email@samplebusiness.com | www.samplebusiness123.com
    """
    
    if len(sys.argv) > 1:
        # If user provides a text file
        with open(sys.argv[1], 'r') as f:
            test_text = f.read()
            
    test_llm(test_text)