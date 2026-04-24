"""该代码由大模型生成，Prompt是：

读取 Excel 的 content 列，多线程调用 DeepSeek，将回复写入新列。
参考api_example.py和api timing_multithread_demo.py代码
"""

from __future__ import annotations
import os
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# —— 在此修改运行参数 ——
INPUT_XLSX = "data/car_review_data.xlsx"
OUTPUT_XLSX = "data/car_review_data_result.xlsx"
CONTENT_COL = "content"
RESULT_COL = "model_response"
MAX_WORKERS = 10


# ===========关于调用大模型API主要代码区域===========

MODEL_NAME = "deepseek-chat"
SYSTEM_PROMPT = "你是一个用户评论分析助手，需要回答这个用户提及了车辆的哪些属性"
USER_PROMPT_TEMPLATE = "用户评论为：{content}"

SYSTEM_PROMPT_new = """你是一个用户评论分析助手，需要回答这个用户提及了车辆的哪些属性
仅需要回复车辆属性，不需要回复其他内容
多个属性用逗号分隔
请根据用户评论，回答用户提及了车辆的哪些属性
"""

def create_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_model(client: OpenAI, user_text: str) -> str:
    user_message = USER_PROMPT_TEMPLATE.format(content=user_text)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_new},
            {"role": "user", "content": user_message},
        ],
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()

# ================================================


def main() -> None:
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

    tasks: list[tuple[Any, str]] = []
    for idx, val in df[CONTENT_COL].items():
        if pd.isna(val) or str(val).strip() == "":
            continue
        tasks.append((idx, str(val).strip()))

    if not tasks:
        print("没有需要处理的非空 content 行。")
        df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
        return

    client = create_client()
    results: dict[Any, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(call_model, client, text): idx for idx, text in tasks
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(tasks),
            desc="API 请求",
            unit="条",
        ):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # noqa: BLE001
                results[idx] = f"[错误] {e}"

    for idx, text in results.items():
        df.loc[idx, RESULT_COL] = text

    df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"完成：共 {len(results)} 行，已保存到 {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
