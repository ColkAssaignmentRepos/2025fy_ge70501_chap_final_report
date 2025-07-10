import math
from collections import defaultdict
from typing import Dict, List, Literal

import numpy as np
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from src.database import get_db_session
from src.indexing import tokenize
from src.models import Document, DocumentFrequency, InvertedIndex, SystemStats, Word

SearchModel = Literal["tfidf", "bm25"]


class SearchResult(BaseModel):
    title: str
    url: str
    score: float
    snippet: str
    content: str  # For PRF


def create_snippet(content: str, query_terms: List[str], max_len: int = 100) -> str:
    """
    本文とクエリ語からスニペットを生成し、クエリ語をハイライトする。
    クエリ語が最も多く出現する文、またはクエリ語周辺のテキストを抽出する。
    """
    # クエリ語のいずれかを含む最初の文を探す
    first_match_pos = -1
    for term in query_terms:
        pos = content.find(term)
        if pos != -1:
            if first_match_pos == -1 or pos < first_match_pos:
                first_match_pos = pos

    if first_match_pos != -1:
        # マッチした位置の前後からテキストを切り出す
        start = max(0, first_match_pos - max_len // 2)
        end = min(len(content), first_match_pos + max_len // 2)
        snippet = content[start:end]
    else:
        # マッチしなかった場合は、文章の先頭から切り出す
        snippet = content[:max_len]

    if len(snippet) > max_len:
        snippet += "..."

    # クエリ語をハイライト
    for term in set(query_terms):  # 重複を除いてハイライト
        snippet = snippet.replace(term, f"<b>{term}</b>")

    return snippet


def _get_candidate_docs(db: Session, query_word_ids: List[int]) -> List[int]:
    """クエリ語を1つでも含む候補文書のIDリストを取得する"""
    stmt = (
        select(Document.id)
        .join(InvertedIndex)
        .filter(InvertedIndex.word_id.in_(query_word_ids))
        .distinct()
    )
    return list(db.execute(stmt).scalars().all())


def _search_tfidf(
    db: Session,
    query_tokens: List[str],
    total_docs: int,
) -> Dict[int, float]:
    """TF-IDFとコサイン類似度に基づいてスコアを計算する"""
    # 1. クエリベクトルを作成
    query_tf: Dict[str, int] = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    query_vector: Dict[int, float] = {}
    query_word_ids: List[int] = []

    stmt = (
        select(Word)
        .filter(Word.term.in_(query_tf.keys()))
        .options(joinedload(Word.doc_frequency))
    )
    words_in_db = db.execute(stmt).scalars().all()
    word_map = {word.term: word for word in words_in_db}

    for term, tf in query_tf.items():
        word_obj = word_map.get(term)
        if word_obj and word_obj.doc_frequency:
            df = word_obj.doc_frequency.doc_frequency
            idf = math.log(total_docs / df) if df > 0 else 0
            query_vector[word_obj.id] = tf * idf
            query_word_ids.append(word_obj.id)

    if not query_vector:
        return {}

    # 2. 候補文書を取得
    candidate_doc_ids = _get_candidate_docs(db, query_word_ids)
    if not candidate_doc_ids:
        return {}

    # 3. 候補文書のTF-IDFベクトルとノルムを計算
    doc_vectors: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    doc_norms: Dict[int, float] = defaultdict(float)

    stmt_ii_query = (
        select(
            InvertedIndex.document_id,
            InvertedIndex.word_id,
            InvertedIndex.term_frequency,
            DocumentFrequency.doc_frequency,
        )
        .join(DocumentFrequency, InvertedIndex.word_id == DocumentFrequency.word_id)
        .filter(InvertedIndex.document_id.in_(candidate_doc_ids))
        .filter(InvertedIndex.word_id.in_(query_word_ids))
    )
    for doc_id, word_id, tf, df in db.execute(stmt_ii_query).all():
        idf = math.log(total_docs / df) if df > 0 else 0
        doc_vectors[doc_id][word_id] = tf * idf

    stmt_ii_all = (
        select(
            InvertedIndex.document_id,
            InvertedIndex.term_frequency,
            DocumentFrequency.doc_frequency,
        )
        .join(DocumentFrequency, InvertedIndex.word_id == DocumentFrequency.word_id)
        .filter(InvertedIndex.document_id.in_(candidate_doc_ids))
    )
    doc_tfidf_squares: Dict[int, float] = defaultdict(float)
    for doc_id, tf, df in db.execute(stmt_ii_all).all():
        idf = math.log(total_docs / df) if df > 0 else 0
        doc_tfidf_squares[doc_id] += (tf * idf) ** 2
    for doc_id, sum_of_squares in doc_tfidf_squares.items():
        doc_norms[doc_id] = math.sqrt(sum_of_squares)

    # 4. コサイン類似度を計算
    scores: Dict[int, float] = {}
    q_vec_np = np.array(list(query_vector.values()))
    q_norm = np.linalg.norm(q_vec_np)
    if q_norm == 0:
        return {}

    for doc_id, vector in doc_vectors.items():
        dot_product = sum(
            query_vector.get(word_id, 0) * tf_idf for word_id, tf_idf in vector.items()
        )
        d_norm = doc_norms.get(doc_id, 1.0)
        if d_norm > 0:
            scores[doc_id] = float(dot_product / (q_norm * d_norm))
    return scores


def _search_bm25(
    db: Session,
    query_tokens: List[str],
    total_docs: int,
    avg_doc_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> Dict[int, float]:
    """BM25アルゴリズムに基づいてスコアを計算する"""
    query_tf: Dict[str, int] = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    stmt = (
        select(Word)
        .filter(Word.term.in_(query_tf.keys()))
        .options(joinedload(Word.doc_frequency))
    )
    words_in_db = db.execute(stmt).scalars().all()
    word_map = {word.term: word for word in words_in_db}
    query_word_ids = [word.id for word in words_in_db if word.id]

    if not query_word_ids:
        return {}

    # IDFの計算
    idfs: Dict[int, float] = {}
    for term, word_obj in word_map.items():
        if word_obj.doc_frequency:
            df = word_obj.doc_frequency.doc_frequency
            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
            idfs[word_obj.id] = idf

    # 候補文書を取得
    candidate_doc_ids = _get_candidate_docs(db, query_word_ids)
    if not candidate_doc_ids:
        return {}

    # 文書長を取得
    doc_lengths_rows = db.execute(
        select(Document.id, Document.token_count).filter(
            Document.id.in_(candidate_doc_ids)
        )
    ).all()
    doc_lengths: Dict[int, int] = {
        doc_id: token_count for doc_id, token_count in doc_lengths_rows
    }

    # BM25スコアを計算
    scores: Dict[int, float] = defaultdict(float)
    stmt_ii = (
        select(
            InvertedIndex.document_id,
            InvertedIndex.word_id,
            InvertedIndex.term_frequency,
        )
        .filter(InvertedIndex.document_id.in_(candidate_doc_ids))
        .filter(InvertedIndex.word_id.in_(query_word_ids))
    )

    for doc_id, word_id, tf in db.execute(stmt_ii).all():
        doc_len = doc_lengths.get(doc_id, 0)
        idf = idfs.get(word_id, 0)

        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))

        scores[doc_id] += idf * (numerator / denominator)

    return scores


def _pseudo_relevance_feedback(
    initial_results: List[SearchResult],
    original_query_tokens: List[str],
    top_n_docs: int = 3,
    top_k_terms: int = 5,
) -> List[str]:
    """擬似適合性フィードバックを実行し、拡張されたクエリトークンリストを返す"""
    if not initial_results:
        return original_query_tokens

    term_scores: Dict[str, float] = defaultdict(float)

    for result in initial_results[:top_n_docs]:
        doc_tokens = tokenize(result.content)
        for token in doc_tokens:
            # 簡単なスコアリング: ここでは出現回数をスコアとする
            # 本来はTF-IDFなどで重み付けする方が良い
            term_scores[token] += 1

    # 元のクエリ語は除外
    for token in original_query_tokens:
        if token in term_scores:
            del term_scores[token]

    # スコア上位の単語を抽出
    expanded_terms = sorted(
        term_scores.items(), key=lambda item: item[1], reverse=True
    )[:top_k_terms]

    new_query_tokens = original_query_tokens + [term for term, score in expanded_terms]
    print(f"[Info] Expanded query with PRF: {new_query_tokens}")
    return new_query_tokens


def search(
    query: str,
    top_k: int = 10,
    model: SearchModel = "bm25",
    use_prf: bool = False,
) -> List[SearchResult]:
    """
    文書を検索し、ランキング付けして返す。
    TF-IDFまたはBM25モデルを使用でき、擬似適合性フィー���バックを適用可能。
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    with get_db_session() as db:
        stats = db.execute(select(SystemStats)).scalar_one_or_none()
        if not stats:
            print(
                "エラー: システム統計情報が見つかりません。'index'コマンドを実行してください。"
            )
            return []
        total_docs = stats.total_documents
        avg_doc_len = stats.average_doc_length

        # --- 擬似適合性フィードバック (PRF) ---
        if use_prf:
            # PRFのために、まず初期検索を行う
            initial_scores = (
                _search_bm25(db, query_tokens, total_docs, avg_doc_len)
                if model == "bm25"
                else _search_tfidf(db, query_tokens, total_docs)
            )

            # 初期検索結果を一時的に取得
            initial_sorted_docs = sorted(
                initial_scores.items(), key=lambda item: item[1], reverse=True
            )[:5]
            initial_doc_ids = [doc_id for doc_id, score in initial_sorted_docs]
            initial_doc_details = (
                db.execute(select(Document).filter(Document.id.in_(initial_doc_ids)))
                .scalars()
                .all()
            )
            initial_doc_map = {doc.id: doc for doc in initial_doc_details}

            initial_results_for_prf = [
                SearchResult(
                    title=doc.title,
                    url=doc.url,
                    score=initial_scores[doc.id],
                    snippet="",  # PRFでは不要
                    content=doc.content,
                )
                for doc_id, score in initial_sorted_docs
                if (doc := initial_doc_map.get(doc_id))
            ]

            query_tokens = _pseudo_relevance_feedback(
                initial_results_for_prf, query_tokens
            )

        # --- メインの検索 ---
        scores: Dict[int, float]
        if model == "bm25":
            scores = _search_bm25(db, query_tokens, total_docs, avg_doc_len)
        else:  # tfidf
            scores = _search_tfidf(db, query_tokens, total_docs)

        if not scores:
            return []

        # スコアでソートし、上位k件を取得
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :top_k
        ]
        if not sorted_docs:
            return []

        # 結果を整形
        results: List[SearchResult] = []
        top_doc_ids = [doc_id for doc_id, score in sorted_docs]

        doc_details_stmt = select(Document).filter(Document.id.in_(top_doc_ids))
        doc_details = db.execute(doc_details_stmt).scalars().all()
        doc_map = {doc.id: doc for doc in doc_details}

        for doc_id, score in sorted_docs:
            doc = doc_map.get(doc_id)
            if doc:
                snippet = create_snippet(doc.content, query_tokens)
                results.append(
                    SearchResult(
                        title=doc.title,
                        url=doc.url,
                        score=score,
                        snippet=snippet,
                        content=doc.content,  # 評価などで再利用するため
                    )
                )
        return results


if __name__ == "__main__":
    sample_query = "プロ野球"
    print(f"--- Searching for: '{sample_query}' (BM25) ---")
    search_results_bm25 = search(sample_query, model="bm25")
    if search_results_bm25:
        for res in search_results_bm25:
            print(f"Score: {res.score:.4f}, Title: {res.title}")
    else:
        print("No results found.")

    print(f"\n--- Searching for: '{sample_query}' (TF-IDF) ---")
    search_results_tfidf = search(sample_query, model="tfidf")
    if search_results_tfidf:
        for res in search_results_tfidf:
            print(f"Score: {res.score:.4f}, Title: {res.title}")
    else:
        print("No results found.")

    print(f"\n--- Searching for: '{sample_query}' (BM25 + PRF) ---")
    search_results_prf = search(sample_query, model="bm25", use_prf=True)
    if search_results_prf:
        for res in search_results_prf:
            print(f"Score: {res.score:.4f}, Title: {res.title}")
    else:
        print("No results found.")
