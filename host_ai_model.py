import os
from openai import OpenAI

# You can set your token like this for testing (or export HF_TOKEN in env)
HF_TOKEN = "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"  # Replace this with your real token

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b:novita",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message.content)
