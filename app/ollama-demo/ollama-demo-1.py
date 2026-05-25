# 使用 LangChain 0.x 的语法访问 Ollama 本地大模型
from langchain_ollama import ChatOllama

def main():
    # 初始化 ChatOllama，指定你运行的模型名称
    llm = ChatOllama(
        model="deepseek-r1:latest",  # 与 ollama run 后的名称保持一致
        base_url="http://localhost:11434",
        temperature=0.1,  # 可选：控制输出随机性
        num_predict=512,  # 可选：最大生成 token 数
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    # 发送消息并获取回复
    response1 = llm.stream(messages)
    response2 = llm.stream("来一段宋词")

    # 打印回复内容
    for chunk in response1:
        print(chunk.content, end="", flush=True)

    for chunk in response2:
        print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    main()