import os
from openai import OpenAI


def main() -> None:
    client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")

    # 组1：翻译风格对比（只改 system，user 不变）
    print("=" * 70)
    print("【组1：翻译风格对比】")
    group_1_user = "请把这句话翻译成英文：这款手机续航很强，但是拍照表现一般。"
    print(f"User 提示词（任务）: {group_1_user}\n")

    print("-" * 70)
    print("[对比 1] System: 你是一个专业的翻译专家，追求准确、正式、书面表达。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的翻译专家，追求准确、正式、书面表达。"},
            {"role": "user", "content": group_1_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    print("-" * 70)
    print("[对比 2] System: 你是一个社交媒体博主，翻译时要口语化、自然、活泼。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个社交媒体博主，翻译时要口语化、自然、活泼。"},
            {"role": "user", "content": group_1_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    print("-" * 70)
    print("[对比 3] System: 你是一个技术产品评测编辑，翻译要简洁并保留产品评价语气。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个技术产品评测编辑，翻译要简洁并保留产品评价语气。"},
            {"role": "user", "content": group_1_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    # 组2：客服回复风格对比（只改 system，user 不变）
    print("=" * 70)
    print("【组2：客服回复风格对比】")
    group_2_user = (
        "用户说：你们物流太慢了，我等了10天才收到，而且包装有破损。"
        "请帮我回复用户。"
    )
    print(f"User 提示词（任务）: {group_2_user}\n")

    print("-" * 70)
    print("[对比 1] System: 你是一个友善且有同理心的客服专员。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个友善且有同理心的客服专员。"},
            {"role": "user", "content": group_2_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    print("-" * 70)
    print("[对比 2] System: 你是一个严格按流程办事的售后经理，回复要专业且有条款感。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一个严格按流程办事的售后经理，回复要专业且有条款感。",
            },
            {"role": "user", "content": group_2_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    print("-" * 70)
    print("[对比 3] System: 你是一个高效的客服机器人，回复要简短，控制在50字以内。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个高效的客服机器人，回复要简短，控制在50字以内。"},
            {"role": "user", "content": group_2_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    # 组3：输出形式对比（只改 system，user 不变）
    print("=" * 70)
    print("【组3：输出形式对比】")
    group_3_user = (
        "从这段话提取关键信息："
        "“昨晚8点我在滨湖区门店下单了一台空气净化器，"
        "今天中午收到货后发现外壳有划痕，订单号A20260423001。”"
    )
    print(f"User 提示词（任务）: {group_3_user}\n")

    print("-" * 70)
    print("[对比 1] System: 你是一个信息抽取助手，只输出一段自然语言总结。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个信息抽取助手，只输出一段自然语言总结。"},
            {"role": "user", "content": group_3_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    print("-" * 70)
    print(
        "[对比 2] System: 你是一个数据标注员，只输出 JSON，字段包括 time、location、item、issue、order_id。"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一个数据标注员，只输出 JSON，字段包括 time、location、item、issue、order_id。",
            },
            {"role": "user", "content": group_3_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()

    print("-" * 70)
    print("[对比 3] System: 你是一个运营分析师，先给一句总结，再给3条可执行建议。")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个运营分析师，先给一句总结，再给3条可执行建议。"},
            {"role": "user", "content": group_3_user},
        ],
        temperature=0.7,
        stream=False,
    )
    print("模型输出:")
    print(response.choices[0].message.content.strip())
    print()


if __name__ == "__main__":
    main()
