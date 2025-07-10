import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import MeCab  # type: ignore
import unidic_lite  # type: ignore
from sqlalchemy import insert, select
from tqdm import tqdm

from src.database import engine, get_db_session
from src.models import (
    Base,
    Document,
    DocumentFrequency,
    InvertedIndex,
    SystemStats,
    Word,
)

# MeCab Taggerのセットアップ
unidic_path = unidic_lite.DICDIR
mecab_tagger = MeCab.Tagger(f"-d {unidic_path}")

# 定数
STOPWORD_POS: Set[str] = {"助詞", "助動詞", "記号", "補助記号"}
DATA_DIR = "data/text"
DOWNLOAD_URL_ENV = "LDCC_DOWNLOAD_URL"


def check_dataset_exists() -> bool:
    """データセットディレクトリの存在を確認する"""
    return os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR)


def extract_text_from_file(filepath: str) -> Tuple[str, str, str]:
    """livedoorニュースコ��パスのファイルからURL, タイトル, 本文を抽出する"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        url = lines[0].strip()
        title = lines[2].strip()
        content = "".join(lines[3:]).strip()
    return url, title, content


def tokenize(text: str) -> List[str]:
    """MeCabを使ってテキストを形態素解析し、ストップワードを除去した単語リストを返す"""
    tokens: List[str] = []
    node = mecab_tagger.parseToNode(text)
    while node:
        features = node.feature.split(",")
        pos = features[0]
        if pos not in STOPWORD_POS:
            # 現代の検索エンジンでは、未知語や固有名詞をそのままの形で保持することが多いため、
            # 無理に見出し語化せず、原型が存在しない場合は表層形を用いる
            base_form = (
                features[6]
                if len(features) > 6 and features[6] != "*"
                else node.surface
            )
            if base_form:
                tokens.append(base_form)
        node = node.next
    return tokens


def run_indexing() -> None:
    """
    data/ディレクトリ内のファイルを処理し、転置索引を構築してDBに保存する。
    既存のデータはすべて削除される。
    """
    if not check_dataset_exists():
        download_url = os.getenv(DOWNLOAD_URL_ENV, "（環境変数が設定されていません）")
        print(f"エラー: データセットが '{DATA_DIR}' に見つかりません。")
        print("1. livedoorニュースコーパスをダウンロードしてください。")
        if download_url != "（環境変数が設定されていません）":
            print(f"   URL: {download_url}")
        else:
            print(
                f"   環境変数 {DOWNLOAD_URL_ENV} にダウンロードURLを設定してください。"
            )
            print("   例: https://www.rondhuit.com/download/ldcc-20140922.tar.gz")
        print(
            "2. 展開し、'text' ディレクトリを 'data' ディレクトリ内に配置してください。"
        )
        return

    # データベースファイルを物理的に削除して再作成する
    db_dir = "db"
    db_path = os.path.join(db_dir, "database.db")
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    if os.path.exists(db_path):
        print(f"Removing existing database file: {db_path}")
        os.remove(db_path)

    print("Re-creating database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Tables re-created.")

    filepaths = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(DATA_DIR)
        for filename in filenames
        if not filename.startswith(".") and filename.endswith(".txt")
    ]

    if not filepaths:
        print(f"エラー: '{DATA_DIR}' 内にテキストファイルが見つかりませんでした。")
        return

    print(f"Found {len(filepaths)} documents. Starting indexing...")

    documents_to_insert: List[Dict[str, object]] = []
    all_words: Set[str] = set()
    doc_word_freqs: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    processed_urls: Set[str] = set()
    total_token_count = 0

    doc_id_counter = 1
    for filepath in tqdm(filepaths, desc="1/5 Reading files"):
        try:
            url, title, content = extract_text_from_file(filepath)
        except Exception:
            continue

        if not url or not title or url in processed_urls:
            continue

        processed_urls.add(url)
        tokens = tokenize(content)
        token_count = len(tokens)
        total_token_count += token_count

        documents_to_insert.append(
            {
                "id": doc_id_counter,
                "title": title,
                "url": url,
                "content": content,
                "token_count": token_count,
            }
        )

        for token in tokens:
            all_words.add(token)
            doc_word_freqs[doc_id_counter][token] += 1

        doc_id_counter += 1

    total_documents = len(documents_to_insert)
    average_doc_length = (
        total_token_count / total_documents if total_documents > 0 else 0.0
    )

    with get_db_session() as db:
        print("\n2/5 Inserting documents...")
        if documents_to_insert:
            db.execute(insert(Document), documents_to_insert)
        db.commit()

        print("3/5 Inserting words...")
        words_to_insert = [{"term": word} for word in sorted(list(all_words))]
        if words_to_insert:
            db.execute(insert(Word), words_to_insert)
        db.commit()

        word_results = db.execute(select(Word)).scalars().all()
        word_map: Dict[str, int] = {
            word.term: word.id for word in word_results if word.id is not None
        }

        inverted_indices_to_insert: List[Dict[str, int]] = []
        doc_freq_counter: Dict[int, int] = defaultdict(int)

        for doc_id, word_freqs in tqdm(
            doc_word_freqs.items(), desc="4/5 Building index"
        ):
            for term, freq in word_freqs.items():
                word_id = word_map.get(term)
                if word_id:
                    inverted_indices_to_insert.append(
                        {
                            "document_id": doc_id,
                            "word_id": word_id,
                            "term_frequency": freq,
                        }
                    )
                    doc_freq_counter[word_id] += 1

        print("Inserting inverted index...")
        if inverted_indices_to_insert:
            chunk_size = 50000  # 5万件ずつのバッチに分割
            for i in tqdm(
                range(0, len(inverted_indices_to_insert), chunk_size),
                desc="  - Batches",
            ):
                batch = inverted_indices_to_insert[i : i + chunk_size]
                db.execute(insert(InvertedIndex), batch)
        db.commit()

        print("Calculating and inserting document frequencies...")
        df_to_insert = [
            {"word_id": word_id, "doc_frequency": freq}
            for word_id, freq in doc_freq_counter.items()
        ]
        if df_to_insert:
            db.execute(insert(DocumentFrequency), df_to_insert)
        db.commit()

        print("5/5 Inserting system stats...")
        stats_to_insert = {
            "total_documents": total_documents,
            "average_doc_length": average_doc_length,
        }
        db.execute(insert(SystemStats), [stats_to_insert])
        db.commit()

    print("Indexing finished successfully.")
    print(f"  - Total documents: {total_documents}")
    print(f"  - Average document length: {average_doc_length:.2f}")


if __name__ == "__main__":
    run_indexing()
