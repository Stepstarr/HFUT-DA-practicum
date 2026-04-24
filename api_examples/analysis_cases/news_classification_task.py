"""基于标题做分类：多线程调用 DeepSeek，写回预测类别并计算准确率。"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# —— 在此修改运行参数 ——
INPUT_XLSX = "data/news_data.xlsx"
OUTPUT_XLSX = "data/news_data_result.xlsx"
TITLE_COL = "title"
TRUE_LABEL_COL = "label"
PRED_LABEL_COL = "大模型返回结果"
SUCCESS_COL = "分类是否正确"
MAX_WORKERS = 4

MODEL_NAME = "deepseek-chat"
# ====================== 在此修改系统提示词 ======================
# 类别有6个，分别是：政策追踪, 国外科技前沿, 典型案例, 院士专家观点, 电力企业动态, 国内科技前沿
SYSTEM_PROMPT = "你是一个文本分类助手。要对以下标题进行分类xxxx【这只是一个示例，需要替换】"

USER_PROMPT_TEMPLATE = (
"标题：{title}\n"
)
# ====================== 在此修改系统提示词 ======================

def create_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_model(client: OpenAI, title_text: str) -> str:
    user_message = USER_PROMPT_TEMPLATE.format(title=title_text)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")

    if TITLE_COL not in df.columns:
        raise SystemExit(f"未找到标题列：{TITLE_COL}。当前列：{list(df.columns)}")
    if TRUE_LABEL_COL not in df.columns:
        raise SystemExit(f"未找到真实类别列：{TRUE_LABEL_COL}。当前列：{list(df.columns)}")

    if PRED_LABEL_COL not in df.columns:
        df[PRED_LABEL_COL] = pd.NA
    if SUCCESS_COL not in df.columns:
        df[SUCCESS_COL] = pd.NA

    tasks: list[tuple[Any, str]] = []
    for idx, title in df[TITLE_COL].items():
        t = "" if pd.isna(title) else str(title).strip()
        if not t:
            continue
        tasks.append((idx, t))

    if not tasks:
        print("没有可处理的标题数据。")
        df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
        return

    client = create_client()
    results: dict[Any, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {executor.submit(call_model, client, title_text): idx for idx, title_text in tasks}
        for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="分类进度", unit="条"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:  # noqa: BLE001
                results[idx] = f"[错误] {e}"

    for idx, pred in results.items():
        df.loc[idx, PRED_LABEL_COL] = pred
        pred_clean = "" if pd.isna(pred) else str(pred).strip()
        true_clean = "" if pd.isna(df.loc[idx, TRUE_LABEL_COL]) else str(df.loc[idx, TRUE_LABEL_COL]).strip()
        df.loc[idx, SUCCESS_COL] = "正确" if pred_clean.casefold() == true_clean.casefold() else "错误"

    valid_df = df[df[TITLE_COL].apply(lambda x: "" if pd.isna(x) else str(x).strip()) != ""].copy()
    correct_mask = (
        valid_df[PRED_LABEL_COL]
        .apply(lambda x: "" if pd.isna(x) else str(x).strip())
        .str.casefold()
        == valid_df[TRUE_LABEL_COL]
        .apply(lambda x: "" if pd.isna(x) else str(x).strip())
        .str.casefold()
    )
    total = len(valid_df)
    correct = int(correct_mask.sum()) if total > 0 else 0
    accuracy = (correct / total) if total > 0 else 0.0

    df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    print(f"完成：共处理 {len(results)} 条，结果已保存到 {OUTPUT_XLSX}")
    print(f"分类准确率：{accuracy:.2%}（{correct}/{total}）")


if __name__ == "__main__":
    main()
