from app.bailian.common import chat_prompt_template, llm
from langchain_core.tools import Tool

def add(a, b):
    return a + b

# 将 add 转成 longchain 可以理解的 tools
add_tools = Tool.from_function(
    func=add,
    name="add", # 全局唯一工具标识
    description="add two numbers",
)

tool_dict = {
    "add": add
}

# 将大模型和 tool 对象绑定
llm_with_tool = llm.bind_tools([
    add_tools,
])

chain = chat_prompt_template | llm_with_tool

# 调用大模型
resp = chain.invoke(input={"role": "计算", "domain": "数学计算", "question": "100+100=?"})

# print(resp)
# 大模型不会帮我们调用工具，只会告诉我们怎么调用。真正的调用者还是智能体本身
for tool_calls in resp.tool_calls:
    print(tool_calls, end="\n")

    args = tool_calls["args"]
    print(args, end="\n")

    func_name = tool_calls["name"]
    print(func_name, end="\n")

    tool_func = tool_dict[func_name]
    print(tool_func, end="\n")

    # 大模型最后只识别了一个参数，裂开。这个时候有必要给工具函数的入参数添加注释。有了注释，大模型就可以更精准地识别工具。
    tool_content = tool_func(args)
    print(tool_content, end="\n")

