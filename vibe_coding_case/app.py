from __future__ import annotations

import os

from flask import Flask, jsonify, render_template_string, request
from openai import OpenAI

MODEL_NAME = "deepseek-chat"
SYSTEM_PROMPT = """你是一个用户评论分析助手，需要回答这个用户提及了车辆的哪些属性。
仅需要回复车辆属性，不需要回复其他内容。
多个属性用逗号分隔。"""
USER_PROMPT_TEMPLATE = "用户评论为：{content}"

app = Flask(__name__)

PAGE_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>评论属性分析</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      margin: 0;
      padding: 0;
      color: #1f2937;
    }
    .container {
      max-width: 760px;
      margin: 48px auto;
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
      padding: 24px;
    }
    h1 { margin-top: 0; font-size: 24px; }
    p { color: #4b5563; }
    textarea {
      width: 100%;
      min-height: 140px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 12px;
      font-size: 14px;
      box-sizing: border-box;
      resize: vertical;
    }
    button {
      margin-top: 12px;
      background: #2563eb;
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 10px 16px;
      font-size: 14px;
      cursor: pointer;
    }
    button:disabled { opacity: 0.7; cursor: not-allowed; }
    .result {
      margin-top: 20px;
      padding: 12px;
      border-radius: 8px;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      min-height: 48px;
      white-space: pre-wrap;
    }
    .error { color: #b91c1c; }
  </style>
</head>
<body>
  <main class="container">
    <h1>用户评论属性分析</h1>
    <p>输入用户评论，点击分析后返回评论中提及的车辆属性。</p>
    <textarea id="comment" placeholder="例如：这车外观很好看，空间够大，但油耗有点高。"></textarea>
    <br />
    <button id="analyzeBtn">分析</button>
    <div id="result" class="result">分析结果会显示在这里。</div>
  </main>

  <script>
    const btn = document.getElementById("analyzeBtn");
    const commentInput = document.getElementById("comment");
    const resultBox = document.getElementById("result");

    async function analyzeComment() {
      const text = commentInput.value.trim();
      if (!text) {
        resultBox.classList.add("error");
        resultBox.textContent = "请输入评论内容后再分析。";
        return;
      }

      btn.disabled = true;
      resultBox.classList.remove("error");
      resultBox.textContent = "正在分析，请稍候...";

      try {
        const resp = await fetch("/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ comment: text }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.error || "分析失败");
        }
        resultBox.textContent = data.attributes || "未识别到属性。";
      } catch (err) {
        resultBox.classList.add("error");
        resultBox.textContent = "请求失败：" + (err.message || "未知错误");
      } finally {
        btn.disabled = false;
      }
    }

    btn.addEventListener("click", analyzeComment);
  </script>
</body>
</html>
"""


def create_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY 环境变量。")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_model(client: OpenAI, user_text: str) -> str:
    user_message = USER_PROMPT_TEMPLATE.format(content=user_text)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


@app.get("/")
def home():
    return render_template_string(PAGE_HTML)


@app.post("/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    comment = str(payload.get("comment", "")).strip()
    if not comment:
        return jsonify({"error": "comment 不能为空"}), 400

    try:
        client = create_client()
        result = call_model(client, comment)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify({"attributes": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
