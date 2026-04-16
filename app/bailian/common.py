from langchain_openai import ChatOpenAI
from pydantic import  SecretStr
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, ChatMessagePromptTemplate, FewShotPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=SecretStr("sk-9ff5ad22ed1942bd8666cec900777236"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    stream_usage=True,
    # Qwen 在 thinking 模式下不支持 create_agent 触发的 required/object tool_choice
    # 关闭 thinking，确保工具调用参数兼容
    extra_body={"enable_thinking": False},
    # temperature=None,
    # max_tokens=None,
    # timeout=None,
    # reasoning_effort="low",
    # max_retries=2,
    # api_key="...",  # If you prefer to pass api key in directly
    # base_url="...",
    # organization="...",
    # other params...
)

# 创建提示词消息模版
system_message_template = ChatMessagePromptTemplate.from_template(
    template="你是一位{role}专家，擅长回答{domain}领域的问题",
    role="system",
)

human_message_template = ChatMessagePromptTemplate.from_template(
    template="用户问题：{question}",
    role="user",
)

# 创建对话提示词模版
chat_prompt_template = ChatPromptTemplate.from_messages([
    system_message_template,
    human_message_template,
])


class AddInputArgs(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool(
    description="add two numbers",
    args_schema=AddInputArgs,
    return_direct=True,
)
def add(a, b):
    """add two numbers"""
    return a + b


def create_calc_tools():
    return [add]


calc_tools = create_calc_tools()