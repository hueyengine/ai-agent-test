from langchain_openai import ChatOpenAI
from pydantic import  SecretStr
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=SecretStr("sk-9ff5ad22ed1942bd8666cec900777236"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    stream_usage=True,
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

# print(llm)

# 创建对话提示词模版
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}专家，擅长回答{domain}领域的问题"),
    ("user", "用户问题：{question}")
])
# 模版 + 变量 =》提示词
prompt = chat_prompt_template.format_messages(
    role="编程",
    domain="web开发",
    question="如何构建一个基于Vue的前端应用"
)

resp = llm.stream(prompt)

for chunk in resp:
    print(chunk.content, end="")