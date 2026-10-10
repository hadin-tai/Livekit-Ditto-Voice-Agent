from huggingface_hub import InferenceClient
import json

from huggingface_hub.utils import BadRequestError
import os
from dotenv import load_dotenv
load_dotenv()

# Your Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

messages = [
    {
        "role": "user",
        "content": "What is the weather in Mumbai?"
    }
]

try:
    response = client.chat.completions.create(
        # model="meta-llama/Llama-3.1-8B-Instruct",
        # model="Qwen/Qwen2.5-7B-Instruct",
        # model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        model="deepseek-ai/DeepSeek-R1",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("\n================ RESPONSE ================\n")
    print(response)

    print("\n================ MESSAGE ================\n")
    print(response.choices[0].message)

    if getattr(response.choices[0].message, "tool_calls", None):
        print("\n✅ TOOL CALL DETECTED\n")
        for tc in response.choices[0].message.tool_calls:
            print("Function:", tc.function.name)
            print("Arguments:", tc.function.arguments)
    else:
        print("\n❌ NO TOOL CALL RETURNED\n")
        print("Content:")
        print(response.choices[0].message.content)

except BadRequestError as e:
    print("ERROR:")
    print(e)
    print("RESPONSE:")
    print(e.response.text)
