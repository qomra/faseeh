from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# For streaming responses
response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[
        {"role": "user", "content": "ما هو جمع كلمة أباءة؟"}
    ],
    temperature=0.1,
    stream=True
)

# Print the response as it comes in
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)