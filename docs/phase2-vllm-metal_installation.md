https://github.com/vllm-project/vllm-metal

curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

To use vllm, activate the virtual environment:
  source /Users/psun/.venv-vllm-metal/bin/activate

Or add the venv to your PATH:
  export PATH="/Users/psun/.venv-vllm-metal/bin:$PATH"

VLLM_USE_MODELSCOPE	False	Set True to change model registry to https://www.modelscope.cn/

pip install modelscope>=1.18.1

export VLLM_USE_MODELSCOPE=True
export PYTORCH_ENABLE_MPS_FALLBACK=1

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --served-model-name qwen-3b \
    --port 8001 \
    --max-model-len 8192 \
    --trust-remote-code

curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-3b",
    "messages": [
      {"role": "user", "content": "用Python写一个简单的二分查找."}
    ]
  }'
{"id":"chatcmpl-9c63fbdb8992715f","object":"chat.completion","created":1773822560,"model":"qwen-3b","choices":[{"index":0,"message":{"role":"assistant","content":"当然可以！下面是一个使用Python实现的简单二分查找算法。这个算法适用于在有序列表中查找特定元素的情况。\n\n二分查找的基本思想是，通过将搜索区间缩小一半来快速找到目标值。具体步骤如下：\n\n1. 确定搜索区间。\n2. 计算中间位置。\n3. 如果中间位置的值等于目标值，则返回该位置。\n4. 如果中间位置的值大于目标值，则在左半部分继续搜索。\n5. 如果中间位置的值小于目标值，则在右半部分继续搜索。\n6. 如果搜索区间为空，则目标值不在列表中。\n\n下面是具体的代码实现：\n\n```python\ndef binary_search(arr, target):\n    \"\"\"\n    使用二分查找算法在有序数组 arr 中查找 target。\n    \n    :param arr: 需要查找的有序数组\n    :param target: 要查找的目标值\n    :return: 如果找到目标值，返回其索引；否则返回 -1\n    \"\"\"\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        \n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n            \n    return -1\n\n# 示例使用\nif __name__ == \"__main__\":\n    sorted_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n    target_value = 7\n    \n    index = binary_search(sorted_array, target_value)\n    \n    if index != -1:\n        print(f\"找到了目标值 {target_value}，索引为 {index}\")\n    else:\n        print(f\"没有找到目标值 {target_value}\")\n```\n\n这段代码定义了一个名为 `binary_search` 的函数，它接受一个有序数组 `arr` 和一个目标值 `target` 作为参数，并返回目标值在数组中的索引（如果存在）。如果目标值不存在于数组中，则返回 `-1`。\n\n你可以根据需要修改 `sorted_array` 和 `target_value` 来测试不同的输入。","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":38,"total_tokens":509,"completion_tokens":471,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}


curl http://localhost:8001/metrics