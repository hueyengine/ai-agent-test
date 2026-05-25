from langchain_openai import ChatOpenAI
from pydantic import  SecretStr
from langchain.agents import create_agent

def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

llm_model_instance = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=SecretStr("sk-9ff5ad22ed1942bd8666cec900777236"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    stream_usage=True
)

graph = create_agent(
    model=llm_model_instance,
    tools=[check_weather],
    system_prompt="You are a helpful assistant",
)
inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
for chunk in graph.stream(inputs):
    print(chunk)