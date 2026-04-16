from langchain.agents import create_agent
from app.bailian.common import llm, create_calc_tools

agent = create_agent(
    model=llm,              # 对应旧版的 llm
    tools=create_calc_tools(),           # 对应旧版的 tools
    system_prompt="你是一个有帮助的助手", # 可选，对应旧版在 agent_kwargs 中设置的 system_message
    # verbose 功能可通过中间件或直接打印结果实现
)

# 4. 调用智能体
response = agent.invoke({
    "messages": [{"role": "user", "content": "100+100=？"}]
})

# 5. 输出结果
print(response["messages"][-1].content)