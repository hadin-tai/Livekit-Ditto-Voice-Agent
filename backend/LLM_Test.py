import time
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()

# Your Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

# List of models to test
MODELS = [
    # "deepseek-ai/DeepSeek-R1", 
    # "Qwen/Qwen2.5-72B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", # Best reasoning with good enogh speed

    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # No fast no good reasoning
    # "mistralai/Mistral-Nemo-Instruct-2407", # Failed
    # "mistralai/Mixtral-8x7B-Instruct-v0.1", # Failed

    # "meta-llama/Meta-Llama-3.1-8B-Instruct", # Fast response with good enogh reasoning
    # "meta-llama/Llama-3.1-8B-Instruct", # Fast response with good enogh reasoning
    # "Qwen/Qwen2.5-7B-Instruct",
    "zai-org/GLM-5.2",
]

QUESTION = "What is the capital of India?"


def test_model(model_name: str, question: str):
    print(f"\n{'=' * 80}")
    print(f"Testing Model: {model_name}")
    print(f"{'=' * 80}")

    try:
        start_time = time.perf_counter()

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=1024,
            temperature=0.7,
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        answer = response.choices[0].message.content

        print(f"⏱️  Time Taken : {total_time:.2f} seconds")
        print(f"🤖 Response:\n{answer}")

    except Exception as e:
        print(f"❌ Failed")
        print(f"Error: {e}")


if __name__ == "__main__":
    print(f"\nQuestion: {QUESTION}")

    for model in MODELS:
        test_model(model, QUESTION)



############################## INDIVIDUAL MODEL TESTING ##############################

# client = InferenceClient(
#     api_key=HF_TOKEN,
#     provider="auto"  # or "novita", "hyperbolic", etc.
# )

# # start_time = time.perf_counter()

# response = client.chat.completions.create(
#     model="meta-llama/Llama-3.1-8B-Instruct",
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain recursion with a Python example."
#         }
#     ],
#     max_tokens=1024,
#     temperature=0.7,
# )

# # end_time = time.perf_counter()
# # total_time = end_time - start_time
# # print(f"⏱️  Time Taken : {total_time:.2f} seconds")

# print(response.choices[0].message.content)