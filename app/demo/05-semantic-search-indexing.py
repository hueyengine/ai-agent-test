## 建立索引
# 1. 读取 PDF，按照页来管理，Document 对象，List[Document]
# 2. 分割文本，文本段（chunk），Document 对象，List[Document]
# 3. 向量化： 文本段 -> 向量，需要潜入模型来辅助
# 4. 向量库：把多个文本段/向量存到向量库，OK了。

# 1. 读取 PDF，按照页来管理，Document 对象，List[Document]
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/nke-10k-2023.pdf")
docs = loader.load()
# print(len(docs)) # 107
# print(type(docs[0])) # <class 'langchain_core.documents.base.Document'>
# print(docs[0])
# page_content='Table of Contents......' 
# metadata={
#   'producer': 'EDGRpdf Service w/ EO.Pdf 22.0.40.0', 
#   'creator': 'EDGAR Filing HTML Converter', 
#   'creationdate': '2023-07-20T16:22:00-04:00', 
#   'title': '0000320187-23-000039', 
#   'author': 'EDGAR Online, a division of Donnelley Financial Solutions', 
#   'subject': 'Form 10-K filed on 2023-07-20 for the period ending 2023-05-31', 
#   'keywords': '0000320187-23-000039; ; 10-K', 
#   'moddate': '2023-07-20T16:22:08-04:00', 
#   'source': 'data/nke-10k-2023.pdf', 
#   'total_pages': 107, 
#   'page': 0, 
#   'page_label': '1'
# }

# 2. 分割文本，文本段（chunk），Document 对象，List[Document]
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # 每个chunk包含1000个字符
    chunk_overlap=200, # 每个chunk与相邻chunk的overlap包含200个字符
    add_start_index=True # 每个chunk包含开始索引
)
all_splits = text_splitter.split_documents(docs) # List[Document]
# print(len(all_splits)) # 516
# print(type(all_splits[0])) # <class 'langchain_core.documents.base.Document'>
# print(all_splits[0])
# page_content='Table of Contents......' 
# metadata={
#   'producer': 'EDGRpdf Service w/ EO.Pdf 22.0.40.0', 
#   'creator': 'EDGAR Filing HTML Converter', 
#   'creationdate': '2023-07-20T16:22:00-04:00', 
#   'title': '0000320187-23-000039', 
#   'author': 'EDGAR Online, a division of Donnelley Financial Solutions', 
#   'subject': 'Form 10-K filed on 2023-07-20 for the period ending 2023-05-31', 
#   'keywords': '0000320187-23-000039; ; 10-K', 
#   'moddate': '2023-07-20T16:22:08-04:00', 
#   'source': 'data/nke-10k-2023.pdf', 
#   'total_pages': 107, 
#   'page': 0, 
#   'page_label': '1',
#   'start_index': 0
# }

# 3. 向量化： 文本段 -> 向量，需要潜入模型来辅助
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# vector_0 = embeddings.embed_query(all_splits[0].page_content)
# print(len(vector_0)) # 768 向量的长度和模型有关，和文本段长度无关
# print(type(vector_0)) # <class 'list'>
# print(vector_0)
# # [-0.027017357, 0.033666175, -0.19594209, -0.084316194, 0.040514074, -0.030035622, ..., ]

# 4. 文本快/向量存储
from langchain_chroma import Chroma # 旧写法 from langchain_community.vectorstores import Chroma
vector_store = Chroma(
    collection_name="exmaple_collection", # 集合名称
    embedding_function=embeddings, # 向量化函数，用于将文本段转换为向量
    persist_directory="./chroma_langchain_db", # 存储路径
)

ids = vector_store.add_documents(documents=all_splits)
print(len(ids)) # 516
print(type(ids)) # <class 'list'>
print(ids) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]


