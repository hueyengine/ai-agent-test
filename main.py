from langchain_ollama import ChatOllama

def main():
    # 初始化 ChatOllama，指定你运行的模型名称
    llm = ChatOllama(
        model="deepseek-r1:latest",  # 与 ollama run 后的名称保持一致
        temperature=0.7,  # 可选：控制输出随机性
        num_predict=512,  # 可选：最大生成 token 数
    )

    # 发送消息并获取回复
    response = llm.invoke("你好，请介绍一下你自己")

    # 打印回复内容
    print(response.content)


if __name__ == "__main__":
    main()