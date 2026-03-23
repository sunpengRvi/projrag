import grpc
import sys
import os

# 【核心修复】将 proto 目录临时加入系统环境路径，彻底解决 gRPC 导入寻址 Bug
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto'))

# 现在可以直接导入了，不需要加 proto. 前缀
import retrieval_pb2
import retrieval_pb2_grpc

def run_test():
    # 1. 建立与 8002 微服务的内部通信通道
    print("[*] 正在连接内部微服务 localhost:8002...")
    with grpc.insecure_channel('localhost:8002') as channel:
        stub = retrieval_pb2_grpc.DocumentRetrieverStub(channel)
        
        # 2. 构造一个硬核的技术问题
        question = "What happens when you declare a path operation function with async def?"
        print(f"[*] 正在发送测试问题: '{question}'")
        
        request = retrieval_pb2.SearchRequest(query=question, top_k=3)
        
        # 3. 发起 RPC 调用
        response = stub.SearchDocuments(request)
        
        print("\n[+] 收到微服务返回的高相关性文档片段：")
        print("-" * 50)
        for i, chunk in enumerate(response.chunks):
            print(f"【片段 {i+1}】相似度: {chunk.score:.4f} | 来源: {chunk.source}")
            print(f"内容预览: {chunk.content[:150]}...")
            print("-" * 50)

if __name__ == '__main__':
    run_test()