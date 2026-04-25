'''
该代码由大模型生成，Prompt是：
我想造一个  @sft/SFT_Text_Sample.jsonl 格式的数据集
数据来源于car_review_data_result.xlsx
user content为
提取评论中的属性，评论为{content}
role": "assistant", "content": 为表格里的model response
'''

import argparse
import json
from pathlib import Path

import pandas as pd


def excel_to_sft_jsonl(
    input_excel: Path,
    output_jsonl: Path,
    content_col: str = "content",
    response_col: str = "model_response",
    user_template: str = "提取评论中的属性，评论为{content}",
) -> int:
    df = pd.read_excel(input_excel)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            content = row.get(content_col)
            response = row.get(response_col)

            if pd.isna(content) or pd.isna(response):
                continue

            content_text = str(content).strip()
            response_text = str(response).strip()
            if not content_text or not response_text:
                continue

            item = {
                "messages": [
                    {"role": "user", "content": user_template.format(content=content_text)},
                    {"role": "assistant", "content": response_text},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Excel to SFT JSONL format (messages list)."
    )
    parser.add_argument(
        "--input",
        default="data/car_review_data_result2.xlsx",
        help="Input Excel file path.",
    )
    parser.add_argument(
        "--output",
        default="sft/car_review_sft_dataset.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--content-col",
        default="content",
        help="Column name for review content.",
    )
    parser.add_argument(
        "--response-col",
        default="model_response",
        help='Column name for assistant response, e.g. "model_response".',
    )
    parser.add_argument(
        "--user-template",
        default="提取评论中的属性，评论为{content}",
        help='User prompt template, must contain "{content}".',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "{content}" not in args.user_template:
        raise ValueError('`--user-template` must contain "{content}".')

    total = excel_to_sft_jsonl(
        input_excel=Path(args.input),
        output_jsonl=Path(args.output),
        content_col=args.content_col,
        response_col=args.response_col,
        user_template=args.user_template,
    )
    print(f"Done. Processed {total} records -> {args.output}")


if __name__ == "__main__":
    main()
