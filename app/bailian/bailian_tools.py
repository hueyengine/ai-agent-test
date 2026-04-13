from langchain_openai import ChatOpenAI
from pydantic import  SecretStr
from langchain_core.prompts import PromptTemplate

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

# 创建提示词模版
prompt_template = PromptTemplate.from_template("今天{something}真不错")
# 模版 + 变量 =》提示词
prompt = prompt_template.format(something="天气")
print(prompt_template)

resp = llm.stream(prompt)

for chunk in resp:
    print(chunk.content, end="")