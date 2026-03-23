import grpc
from concurrent import futures
import time

# 导入我们之前用 protoc 自动生成的 gRPC 桩代码
#import proto.retrieval_pb2 as retrieval_pb2
#import proto.retrieval_pb2_grpc as retrieval_pb2_grpc

import sys
import os

# 【核心修复】将 proto 目录临时加入系统环境路径，彻底解决 gRPC 导入寻址 Bug
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto'))

# 现在可以直接导入了，不需要加 proto. 前缀
import retrieval_pb2
import retrieval_pb2_grpc

from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

class DocumentRetrieverService(retrieval_pb2_grpc.DocumentRetrieverServicer):
    def __init__(self):
        print("[*] 正在启动企业级 gRPC 检索微服务...")
        
        # 1. 连接 Docker 版 Milvus
        self.milvus_uri = "http://localhost:19530"
        self.collection_name = "fastapi_docs"
        print(f"[*] 正在连接 Milvus: {self.milvus_uri}")
        self.client = MilvusClient(uri=self.milvus_uri)
        
        # 2. 将 BGE-m3 模型加载到常驻内存中 (这样每次检索就只有几毫秒延迟)
        local_bge_path = "./scripts/models/BAAI/bge-m3" 
        print(f"[*] 正在将 BGE-m3 模型加载至内存 (路径: {local_bge_path})...")
        self.embedding_fn = BGEM3EmbeddingFunction(
            model_name=local_bge_path, 
            use_fp16=False
        )
        
        # 3. 必须先将集合加载到 Milvus 的内存中才能进行检索
        self.client.load_collection(self.collection_name)
        print("[+] 模型与数据库加载完毕，微服务准备就绪！")

    def SearchDocuments(self, request, context):
        """实现 .proto 文件中定义的接口逻辑"""
        query = request.query
        top_k = request.top_k
        print(f"\n[>] 收到 gRPC 检索请求 -> Query: '{query}', Top_K: {top_k}")

        start_time = time.time()

        # 1. 实时将用户的提问转成向量 (提取稠密向量)
        # 注意：encode_queries 是专门用来处理简短提问的，它在底层的处理逻辑跟 encode_documents 不同
        query_vector = self.embedding_fn.encode_queries([query])["dense"][0]

        # 2. 在 Milvus 中执行极其快速的 ANN (近似最近邻) 向量检索
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}}, # ef越大召回率越高但略慢
            output_fields=["content", "source"] # 告诉数据库顺便把文本和出处一起返回给我
        )

        # 3. 组装 gRPC 响应体
        response = retrieval_pb2.SearchResponse()
        
        # search_res 是一个二维列表，因为我们只传了一个 query，所以取 [0]
        for hit in search_res[0]:
            chunk = response.chunks.add()
            chunk.content = hit["entity"]["content"]
            chunk.source = hit["entity"]["source"]
            chunk.score = hit["distance"] # 余弦相似度打分

        elapsed = (time.time() - start_time) * 1000
        print(f"[<] 检索完成，耗时: {elapsed:.2f} ms")
        
        return response

def serve():
    """启动 gRPC Server 监听"""
    # 配置线程池处理并发请求
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    retrieval_pb2_grpc.add_DocumentRetrieverServicer_to_server(
        DocumentRetrieverService(), server
    )
    
    # 绑定我们在架构设计中定好的 8002 端口
    server.add_insecure_port('[::]:8002')
    server.start()
    print("[+] gRPC 检索微服务已成功绑定在 0.0.0.0:8002 端口，正在监听内部请求...")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[-] 收到中断信号，正在关闭微服务...")
        server.stop(0)

if __name__ == '__main__':
    serve()