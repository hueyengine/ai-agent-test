from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. 构建嵌入模型
embeddings = OllamaEmbeddings(model="qwen3-embedding:4b")

# 评分方式
score_methods = [
    "default", # 默认评分方式，用两个向量的夹角余弦值度量相似度
    "cosine", # 余弦相似度，用两个向量的夹角余弦值度量相似度
    "l2", # 欧氏距离，用两个向量的欧氏距离度量相似度
    "ip", # 点积，用两个向量的点积度量相似度
    # "euclidean", # 欧氏距离，用两个向量的欧氏距离度量相似度
    # "manhattan", # 曼哈顿距离，用两个向量的曼哈顿距离度量相似度
    # "chebyshev", # 切比雪夫距离，用两个向量的切比雪夫距离度量相似度
    # "hamming", # 汉明距离，用两个向量的汉明距离度量相似度
    # "jaccard", # 杰卡德距离，用两个向量的杰卡德距离度量相似度
    # "dice", # 骰子距离，用两个向量的骰子距离度量相似度
]

# 2. 构建向量库（知识库）和 4 个 collection
persist_dir = "./chroma_score_db"
# vector_store = Chroma(
#     collection_name="exmaple_collection", # 集合名称
#     embedding_function=embeddings, # 向量化函数，用于将文本段转换为向量
#     persist_directory="./chroma_langchain_db", # 存储路径
# )
vector_stores = []
for score_method in score_methods:
    collection_metadata = { "hnsw:space": score_method }
    if score_method == "default":
        collection_metadata = None
    
    collection_name = f"my_collection_{score_method}"
    vector_store = Chroma(
        collection_name=collection_name, # 集合名称
        embedding_function=embeddings, # 向量化函数，用于将文本段转换为向量
        persist_directory=persist_dir, # 存储路径
        collection_metadata=collection_metadata, # 集合元数据
    )
    vector_stores.append(vector_store)

def indexing(docs):
    print("\n加入文档到向量库...")
    for vector_store in vector_stores:
        ids = vector_store.add_documents(docs)
        print(f"\n集合：{vector_store._collection.name}")
        print(ids)
        print("-" * 50)

def query_with_score(query):
    for i in range(len(score_methods)):
        results = vector_stores[i].similarity_search_with_score(query)
        print(f"\n搜索：{query}")
        for doc, score in results:
            print(doc.page_content, end="")
            print(f"{score_methods[i]}: {score}")

docs = [
    Document(page_content="这个小米手机很好用"),
    Document(page_content="我国山西地区生产小米"),
]

indexing(docs)

query_with_score("雷军最近不开心")