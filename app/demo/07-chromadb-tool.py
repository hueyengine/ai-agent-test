import chromadb

# 列出向量库的 collections 和记录
def list_collections(db_path: str):
    client = chromadb.PersistentClient(db_path)
    collections = client.list_collections()
    print(f"chromadb 向量库{db_path}有{len(collections)}个 collection")

    for i, collection in enumerate(collections):
        print(f"collection {i+1}：{collection.name}，共有{collection.count()}条记录")

# 删除 collection 和记录
def delete_collection(db_path: str, collection_name: str):
    try:
        client = chromadb.PersistentClient(db_path)
        client.delete_collection(collection_name)
        print(f"collection {collection_name} 和记录已删除")
    except Exception as e:
        print(f"删除 collection {collection_name} 和记录失败：{e}")

list_collections("./chroma_langchain_db")
