import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "你好，介绍一下你自己"},
    ],
    stream=False
)

print(response.choices[0].message.content)