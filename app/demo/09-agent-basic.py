# langchain 1.x 语法
from langchain.agents import create_agent
from dotenv import load_dotenv
from pathlib import Path

# 从项目根目录的 .env 加载环境变量（与当前工作目录无关）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

agent = create_agent(
    model="deepseek:deepseek-chat"
)

# print(agent)

results = agent.invoke({
    "messages": [
        { "role": "user", "content": "What's the weather in San Francisco?" }
    ]
})

messages = results["messages"]
print(f"历史小时：{len(messages)}条")
for message in messages:
    message.pretty_print()