from langchain_openai import ChatOpenAI
from pydantic import  SecretStr
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, ChatMessagePromptTemplate, FewShotPromptTemplate

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

# 模版 + 变量 =》提示词
prompt = chat_prompt_template.format_messages(
    role="编程",
    domain="Web开发",
    question="你擅长什么？"
)

example_template = "输入：{input}\n输出：{output}"

examples = [
    {"input": "将'Hello'翻译成中文", "output": "你好"},
    {"input": "将'Goodbye'翻译成中文", "output": "再见"},
    {"input": "将'Pen'翻译成中文", "output": "钢笔"},
]

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate.from_template(example_template),
    prefix="请将以下英文翻译成中文：",
    suffix="输入：{text}\n输出：",
    input_variables=["text"],
)

print(few_shot_prompt_template)

prompt = few_shot_prompt_template.format(text="Thank you!")
print(prompt)

resp = llm.stream(prompt)

for chunk in resp:
    print(chunk.content, end="")