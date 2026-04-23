import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def create_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def call_api(client: OpenAI, prompt: str) -> tuple[str, float]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个有用的助手"},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )
    elapsed = time.perf_counter() - start
    return response.choices[0].message.content.strip(), elapsed


def run_sequential(client: OpenAI, prompts: list[str]) -> tuple[float, list[float]]:
    print("\n" + "=" * 70)
    print("串行调用（单线程）")
    print("=" * 70)

    all_start = time.perf_counter()
    per_call_times = []
    for idx, prompt in enumerate(prompts, start=1):
        _, elapsed = call_api(client, prompt)
        per_call_times.append(elapsed)
        print(f"第 {idx} 次调用耗时: {elapsed:.2f}s")
    total_elapsed = time.perf_counter() - all_start
    return total_elapsed, per_call_times


def run_multithread(client: OpenAI, prompts: list[str], max_workers: int) -> tuple[float, list[float]]:
    print("\n" + "=" * 70)
    print(f"并发调用（多线程，线程数={max_workers}）")
    print("=" * 70)

    all_start = time.perf_counter()
    per_call_times = [0.0] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(call_api, client, prompt): idx for idx, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            _, elapsed = future.result()
            per_call_times[idx] = elapsed
            print(f"第 {idx + 1} 次调用耗时: {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - all_start
    return total_elapsed, per_call_times


def print_summary(
    sequential_total: float,
    sequential_times: list[float],
    threaded_total: float,
    threaded_times: list[float],
) -> None:
    print("\n" + "=" * 70)
    print("耗时对比结果")
    print("=" * 70)

    seq_avg = sum(sequential_times) / len(sequential_times)
    th_avg = sum(threaded_times) / len(threaded_times)
    speedup = sequential_total / threaded_total if threaded_total > 0 else 0.0

    print(f"串行总耗时: {sequential_total:.2f}s")
    print(f"并发总耗时: {threaded_total:.2f}s")
    print(f"串行平均单次耗时: {seq_avg:.2f}s")
    print(f"并发平均单次耗时: {th_avg:.2f}s")
    print(f"整体加速比(串行/并发): {speedup:.2f}x")


def main() -> None:
    client = create_client()

    request_count = 6
    max_workers = 3
    prompts = [f"请用一句话介绍机器学习，版本{i}" for i in range(1, request_count + 1)]

    sequential_total, sequential_times = run_sequential(client, prompts)
    threaded_total, threaded_times = run_multithread(client, prompts, max_workers=max_workers)
    print_summary(sequential_total, sequential_times, threaded_total, threaded_times)


if __name__ == "__main__":
    main()
