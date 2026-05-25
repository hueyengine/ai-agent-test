# 使用 LangChain 1.x 的语法访问 Ollama 本地大模型
from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="ollama:deepseek-r1:latest",
    base_url="http://localhost:11434",
    temperature=0.1,
    timeout=30,
    max_tokens=2000,
)

for chunk in model.stream("来一段唐诗"):
    print(chunk.content, end="", flush=True)
