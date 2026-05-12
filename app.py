import os
import gradio as gr
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def chat(message, history):
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=512
    )
    return response.choices[0].message.content

demo = gr.ChatInterface(
    fn=chat,
    title="My AI Chatbot",
    description="Powered by Groq + LLaMA 3.3"
)
demo.launch()
