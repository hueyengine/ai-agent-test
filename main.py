from langchain_ollama import ChatOllama

def main():
    # 初始化 ChatOllama，指定你运行的模型名称
    llm = ChatOllama(
        model="deepseek-r1:latest",  # 与 ollama run 后的名称保持一致
        temperature=0,  # 可选：控制输出随机性
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
    response = llm.stream(messages)

    # 打印回复内容
    for chunk in response:
        print(chunk.content, end="")


if __name__ == "__main__":
    main()