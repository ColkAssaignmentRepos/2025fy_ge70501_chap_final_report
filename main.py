import click

from src.evaluation import run_evaluation
from src.indexing import run_indexing
from src.search import SearchModel, search


@click.group()
def cli() -> None:
    """情報検索システム CLI"""
    pass


@cli.command()
def index() -> None:
    """データから検索用のインデックスを構築します。"""
    click.echo("インデックス構築を開始します...")
    run_indexing()
    click.echo("インデックス構築が完了しました。")


@cli.command("search")
@click.argument("query")
@click.option("--top-k", default=10, help="表示する上位件数")
@click.option(
    "--model",
    type=click.Choice(["tfidf", "bm25"], case_sensitive=False),
    default="bm25",
    help="使用する検索モデル (tfidf or bm25)",
)
@click.option(
    "--prf",
    is_flag=True,
    default=False,
    help="擬似適合性フィードバック(PRF)を有効にする",
)
def search_cli(query: str, top_k: int, model: SearchModel, prf: bool) -> None:
    """指定されたクエリで文書を検索します。"""
    model_name = f"{model.upper()}"
    if prf:
        model_name += " + PRF"
    click.echo(f"'{query}' を上位{top_k}件で検索します... (モデル: {model_name})")

    results = search(query=query, top_k=top_k, model=model, use_prf=prf)

    if not results:
        click.echo("結果が見つかりませんでした。")
        return

    click.echo("\n--- 検索結果 ---")
    for i, res in enumerate(results, 1):
        click.echo(f"{i}. Title: {res.title}")
        click.echo(f"   URL: {res.url}")
        click.echo(f"   Score: {res.score:.4f}")
        # スニペットの<b>タグを除去して表示
        snippet_plain = res.snippet.replace("<b>", "").replace("</b>", "")
        click.echo(f"   Snippet: {snippet_plain}\n")
    click.echo("--------------------")


@cli.command()
@click.option("--top-k", default=10, help="評価に使う上位件数 (Precision@k, Recall@k)")
@click.option(
    "--model",
    type=click.Choice(["tfidf", "bm25"], case_sensitive=False),
    default="bm25",
    help="評価対象の検索モデル (tfidf or bm25)",
)
def evaluate(top_k: int, model: SearchModel) -> None:
    """システムの検索性能を評価します。"""
    click.echo(f"モデル '{model.upper()}' の検索性能を評価します...")
    run_evaluation(k=top_k, model=model)


if __name__ == "__main__":
    cli()
