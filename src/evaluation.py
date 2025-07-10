from typing import Dict, List, Set, Tuple

from src.search import SearchModel, SearchResult, search


def generate_fixed_eval_data() -> Dict[str, Set[str]]:
    """
    BM25モデルの検索結果に基づき、評価用の仮の正解データを生成する。
    これにより、異なるモデルを同じ基準で比較できるようになる。
    """
    print("[Info] Generating fixed evaluation data based on BM25 search results...")
    fixed_eval_data: Dict[str, Set[str]] = {}
    queries = [
        "IT技術と特許",
        "iPhone関連の話題",
        "映画の興行収入",
        "プロ野球の試合結果",
        "グルメとレシピ",
        "夏の甲子園",
        "AIと仕事",
        "円安の影響",
        "最新スマートフォン",
        "健康的な食事",
    ]
    for query in queries:
        # 基準となるBM25モデルで検索
        results = search(query, top_k=5, model="bm25")
        if results:
            fixed_eval_data[query] = {res.url for res in results}
        else:
            fixed_eval_data[query] = set()
    print("[Info] Fixed evaluation data generated.")
    return fixed_eval_data


def calculate_metrics(
    retrieved_results: List[SearchResult], relevant_docs: Set[str], k: int
) -> Tuple[float, float, float]:
    """
    Precision@k, Recall@k, Average Precision (AP)を計算する
    """
    if not retrieved_results and not relevant_docs:
        return (1.0, 1.0, 1.0)  # 両方空なら完璧とみなす
    if not retrieved_results:
        return (0.0, 0.0, 0.0)

    retrieved_urls = [result.url for result in retrieved_results[:k]]
    relevant_retrieved_set = set(retrieved_urls) & relevant_docs

    # Precision@k
    precision_at_k = len(relevant_retrieved_set) / k if k > 0 else 0.0

    # Recall@k
    total_relevant = len(relevant_docs)
    recall_at_k = (
        len(relevant_retrieved_set) / total_relevant if total_relevant > 0 else 0.0
    )

    # Average Precision (AP)
    ap = 0.0
    relevant_count_so_far = 0
    for i, url in enumerate(retrieved_urls):
        if url in relevant_docs:
            relevant_count_so_far += 1
            precision_at_i = relevant_count_so_far / (i + 1)
            ap += precision_at_i

    ap /= total_relevant if total_relevant > 0 else 1

    return precision_at_k, recall_at_k, ap


def run_evaluation(k: int = 10, model: SearchModel = "bm25") -> None:
    """
    評価データセットを用いて指定された検索モデルの性能を評価し、結果を表示する。
    """
    # 固定された評価データを使用
    EVALUATION_DATA = generate_fixed_eval_data()

    print(f"Starting evaluation for model: {model.upper()}...")

    all_precisions: List[float] = []
    all_recalls: List[float] = []
    all_aps: List[float] = []

    for query, relevant_docs in EVALUATION_DATA.items():
        print(f"\n--- Query: '{query}' ---")

        try:
            search_results = search(query, top_k=k, model=model)
            p_at_k, r_at_k, ap = calculate_metrics(search_results, relevant_docs, k)

            print(f"Precision@{k}: {p_at_k:.4f}")
            print(f"Recall@{k}:    {r_at_k:.4f}")
            print(f"Average Precision: {ap:.4f}")

            all_precisions.append(p_at_k)
            all_recalls.append(r_at_k)
            all_aps.append(ap)

        except Exception as e:
            print(f"An error occurred during search for query '{query}': {e}")
            all_precisions.append(0.0)
            all_recalls.append(0.0)
            all_aps.append(0.0)

    # MAP (Mean Average Precision) の計算
    mean_ap = sum(all_aps) / len(all_aps) if all_aps else 0.0
    mean_precision = (
        sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
    )
    mean_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0

    print("\n--- Overall Performance ---")
    print(f"Model: {model.upper()}")
    print(f"Average Precision@{k}: {mean_precision:.4f}")
    print(f"Average Recall@{k}:    {mean_recall:.4f}")
    print(f"Mean Average Precision (MAP): {mean_ap:.4f}")
    print("--------------------------")


if __name__ == "__main__":
    run_evaluation(model="bm25")
    print("\n" + "=" * 40 + "\n")
    run_evaluation(model="tfidf")
