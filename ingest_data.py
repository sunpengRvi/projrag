import os
import glob
from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_text_splitters import MarkdownTextSplitter

class KnowledgeBaseManager:
    def __init__(self, uri="http://localhost:19530", collection_name="fastapi_docs"):
        """初始化知识库管理器"""
        print(f"[*] 正在连接 Milvus 数据库: {uri}...")
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        
        #print("[*] 正在加载 BGE-m3 向量模型 (首次运行会自动下载权重)...")
        # BGE-m3 默认输出 1024 维度的向量
        #self.embedding_fn = BGEM3EmbeddingFunction(use_fp16=False)
        local_bge_path = "./scripts/models/BAAI/bge-m3" 
        print(f"[*] 正在从本地路径加载 BGE-m3: {local_bge_path}")
        
        self.embedding_fn = BGEM3EmbeddingFunction(
            model_name=local_bge_path, 
            use_fp16=False
        )
        self.vector_dim = 1024 
        
        self._setup_collection()

    def _setup_collection(self):
        """企业级实践：定义严谨的 Schema，而不是使用默认的动态 Schema"""
        if self.client.has_collection(self.collection_name):
            print(f"[*] 集合 '{self.collection_name}' 已存在，准备清空重建...")
            self.client.drop_collection(self.collection_name)
            
        print(f"[*] 正在创建强类型 Schema 集合: {self.collection_name}...")
        schema = MilvusClient.create_schema(
            auto_id=True, # 自动生成主键
            enable_dynamic_field=False
        )
        # 定义主键
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # 定义文本内容 (设置最大长度)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        # 定义来源标识
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=1024)
        # 定义向量字段 (必须与 BGE-m3 维度一致)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_dim)

        # 设置索引参数 (HNSW 算法适合亿级数据高并发检索)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            metric_type="COSINE", # 使用余弦相似度
            index_type="HNSW",
            params={"M": 16, "efConstruction": 200}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        print("[+] 集合创建成功！")

    def process_and_insert(self, docs_dir="./data"):
        """读取 Markdown、切分并批量存入 Milvus"""
        # 1. 寻找所有 md 文件
        md_files = glob.glob(os.path.join(docs_dir, "*.md"))
        if not md_files:
            print(f"[-] 在 {docs_dir} 目录下没有找到 Markdown 文件！")
            return

        # 2. 使用 LangChain 进行智能 Markdown 切分
        # 它会尽量保持标题和正文、代码块的连贯性
        splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        
        all_chunks = []
        all_sources = []
        
        print(f"[*] 开始解析并切分 {len(md_files)} 个文档...")
        for file_path in md_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            # 记录这段文字来源于哪个文件
            all_sources.extend([os.path.basename(file_path)] * len(chunks))

        print(f"[*] 共切分为 {len(all_chunks)} 个片段。正在生成向量 (可能需要一些时间)...")
        # 3. 批量生成向量 
        # BGE-m3 返回的是一个字典: {'dense': [...], 'sparse': [...], 'colbert_vecs': [...]}
        embeddings_dict = self.embedding_fn.encode_documents(all_chunks)
        dense_vectors = embeddings_dict["dense"]
        
        # 4. 组装数据并存入 Milvus
        data_to_insert = []
        for i in range(len(all_chunks)):
            data_to_insert.append({
                "content": all_chunks[i],
                "source": all_sources[i],
                "vector": dense_vectors[i]  # 提取纯稠密向量存入 HNSW 索引
            })

        print(f"[*] 正在将 {len(data_to_insert)} 条数据批量插入 Milvus...")
        res = self.client.insert(
            collection_name=self.collection_name,
            data=data_to_insert
        )
        print(f"[+] 入库完成！成功插入 {res['insert_count']} 条数据。")

if __name__ == "__main__":
    # 实例化并运行流水线
    manager = KnowledgeBaseManager()
    manager.process_and_insert()