# langchain 1.x 语法
from langchain.agents import create_agent
from dotenv import load_dotenv
from pathlib import Path

# 从项目根目录的 .env 加载环境变量（与当前工作目录无关）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[get_weather]
)

print(agent.nodes)
# {
#   '__start__': <langgraph.pregel._read.PregelNode object at 0x106f86ba0>, 
#   'model': <langgraph.pregel._read.PregelNode object at 0x10706d1d0>, 
#   'tools': <langgraph.pregel._read.PregelNode object at 0x10706d450>
# }

results = agent.invoke({
    "messages": [
        { "role": "user", "content": "What's the weather in San Francisco?" }
    ]
})

messages = results["messages"]
print(f"历史小时：{len(messages)}条")
for message in messages:
    message.pretty_print()