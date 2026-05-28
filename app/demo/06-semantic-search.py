from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma # 旧写法 from langchain_community.vectorstores import Chroma

# 1. 构建嵌入模型
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. 构建向量库（知识库）
vector_store = Chroma(
    collection_name="exmaple_collection", # 集合名称
    embedding_function=embeddings, # 向量化函数，用于将文本段转换为向量
    persist_directory="./chroma_langchain_db", # 存储路径
)

# 3. 查询
# 查询有好几种方式
# 3.1 相似度查询
results = vector_store.similarity_search(
    query="What is the company's business?", # 查询语句
    k=3, # 返回结果数量
)
for (index, result) in enumerate(results): # results 是列表，每个元素是一个 Document 对象
    print(index)
    print(result.page_content[:100])

print("-" * 50)

# 3.2 带分数的相似度查询
results = vector_store.similarity_search_with_score(
    query="What is the company's business?", # 查询语句
    k=3, # 返回结果数量
)
for (doc, score) in results: # results 是元组列表，每个元组包含一个 Document 对象和一个分数
    print(score)
    print(doc.page_content[:100])

print("-" * 50)

# 3.3 用向量进行相似度查询
vector = embeddings.embed_query("What is the company's business?")
results = vector_store.similarity_search_by_vector(
    vector, # 查询向量
    k=3, # 返回结果数量
)
for (index, result) in enumerate(results): # results 是列表，每个元素是一个 Document 对象
    print(index)
    print(result.page_content[:100])

print("-" * 50)

# chain：langchain：大模型，提示词模版，tools，output，Runable
# 用检索器进行相似度查询
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=3)

results = retriever.invoke("What is the company's business?")
for (index, result) in enumerate(results): # results 是列表，每个元素是一个 Document 对象
    print(index)
    print(result.page_content[:100])