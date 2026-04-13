import sys
from openai import OpenAI

# Point the client to your local vLLM server
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY" # vLLM does not require an API key by default
)

MODEL_NAME = "Qwen2.5-1.5B"

def main():
    print(f"--- Connected to local vLLM ({MODEL_NAME}) ---")
    print("Type 'quit' or 'exit' to end the session.\n")

    # Initialize the conversation history with a system prompt
    messages = [
        {"role": "system", "content": "You are an AI assistant. Do not respond to me in more than 40 words."}
    ]

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting chat...")
                break
            if not user_input.strip():
                continue

            # Add the user's message to the history
            messages.append({"role": "user", "content": user_input})

            print("Qwen: ", end="", flush=True)
            
            # Call the API with stream=True
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1, # Low temperature for more deterministic extraction tasks
                stream=True
            )

            assistant_reply = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_reply += content
            
            print() # Print a newline after the full response

            # Save the assistant's reply to the history to maintain context
            messages.append({"role": "assistant", "content": assistant_reply})

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\n[Error communicating with server]: {e}")
            # Remove the last user message so they can try again without breaking history
            messages.pop() 

if __name__ == "__main__":
    main()

