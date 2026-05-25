"""
使用 DeepSeek 大模型访问 DeepSeek API
"""

from pathlib import Path
from dotenv import load_dotenv
# 旧版代码，LangChain 0.x 的语法
from langchain_deepseek import ChatDeepSeek

# 从项目根目录的 .env 加载环境变量（与当前工作目录无关）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

model = ChatDeepSeek(
    model="deepseek-chat", # 默认使用 deepseek-chat 模型，可选 deepseek-r1, deepseek-reasoner
    temperature=0.1, # 温度
    max_tokens=2000, # 最大 token 数
    timeout=30, # 超时时间
    max_retries=3, # 最大重试次数
)

response = model.stream("来一段毛泽东诗词")

for chunk in response:
    print(chunk.content, end="", flush=True)