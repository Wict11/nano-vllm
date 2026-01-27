import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        # 格式化为聊天模型的输入格式，加上start和end generation提示
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, # 不进行tokenization返回token id列表，而是直接返回字符串
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    # 生成文本
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
