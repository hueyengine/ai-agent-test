from langchain.agents import create_agent
from app.bailian.common import llm, create_calc_tools, chat_prompt_template
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

# 1. 定义 Pydantic models 的 Schema type
class Summary(BaseModel):
    """计算结果的结构化输出"""
    args: str = Field(description="工具的入参")
    result: int = Field(description="计算的结果")

class UserInfo(BaseModel):
    name: str
    age: int
    city: str
# --------------------------
# 2. 官网 create_agent + 4 种正确写法
# 完全对应：
# response_format: Union[ToolStrategy, ProviderStrategy, type[SchemaT], None]
# --------------------------

# 方式1：ToolStrategy（强制工具策略）
agent1_1 = create_agent(
    model=llm,
    tools=create_calc_tools(),
    response_format=ToolStrategy(Summary),
)

# 方式2：ProviderStrategy（强制原生策略）
agent2_1 = create_agent(
    model=llm,
    tools=create_calc_tools(),
    response_format=ProviderStrategy(Summary, strict=True),
)

# 方式3：直接传类（自动策略 → 最常用）
agent3_1 = create_agent(
    model=llm,
    tools=create_calc_tools(),
    response_format=Summary,  # 计算结构
)
agent3_2 = create_agent(
    model=llm,
    tools=[],  # 提取信息不需要工具，这里是空列表
    response_format=UserInfo,  # 用户信息结构
)

# 方式4：None（关闭结构化）
agent4_1 = create_agent(
    model=llm,
    tools=create_calc_tools(),
    response_format=None,
)

# 3. 使用 chat_prompt_template 生成提示词
question = "100 + 100 = ?"

# --------------------------
# 3. 调用智能体（保持与 Summary 结构一致）
# --------------------------
# ✅ 计算：用 Summary 结构的 agent3_1
response1 = agent3_1.invoke({
    "messages": [{"role": "user", "content": "请使用工具计算 100 + 100"}],
})

# ✅ 提取信息：用 UserInfo 结构的 agent3_2
response2 = agent3_2.invoke({
    "messages": [{"role": "user", "content": "我叫张三，今年25岁，住在上海"}],
})

# --------------------------
# 正确打印结果
# --------------------------
print("==== 计算结果（response1）====")
print(response1)  # 结构化输出

print("\n==== 用户信息（response2）====")
print(response2["structured_response"])