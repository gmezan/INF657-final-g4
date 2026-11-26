"""Index corpus_documentos.csv into a local Qdrant vector database.

Usage:
    uv run index_corpus.py                 # index with defaults
    uv run index_corpus.py --recreate      # drop and rebuild the collection
    uv run index_corpus.py --query "¿Cuántas semillas hay en ER-000?"
"""

import argparse
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / "corpus_documentos.csv"
DB_PATH = ROOT / "qdrant_data"
COLLECTION = "corpus"
MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64

# E5 models are trained with these prefixes; omitting them hurts retrieval.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def load_corpus(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"doc_id": str, "site": str})
    df["year"] = df["year"].astype(int)
    df["reliability"] = df["reliability"].astype(float)
    return df


def build_index(client: QdrantClient, model: SentenceTransformer, df: pd.DataFrame, recreate: bool) -> None:
    dim = model.get_embedding_dimension()

    if client.collection_exists(COLLECTION):
        if not recreate:
            print(f"Collection '{COLLECTION}' already exists ({client.count(COLLECTION).count} points). "
                  "Use --recreate to rebuild.")
            return
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    # Local mode filters payloads without explicit indexes (payload indexes only apply to server Qdrant).

    texts = [PASSAGE_PREFIX + t for t in df["text"]]
    vectors = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True)

    points = [
        models.PointStruct(id=i, vector=vec.tolist(), payload=row)
        for i, (vec, row) in enumerate(zip(vectors, df.to_dict(orient="records")))
    ]
    client.upload_points(COLLECTION, points=points, batch_size=256)
    print(f"Indexed {client.count(COLLECTION).count} documents into '{COLLECTION}' at {DB_PATH}")


def search(client: QdrantClient, model: SentenceTransformer, query: str, k: int = 5, site: str | None = None):
    qvec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    flt = None
    if site:
        flt = models.Filter(must=[models.FieldCondition(key="site", match=models.MatchValue(value=site))])
    return client.query_points(COLLECTION, query=qvec.tolist(), limit=k, query_filter=flt, with_payload=True).points


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    ap.add_argument("--query", help="run a test query after indexing")
    ap.add_argument("--site", help="optional site filter for --query (e.g. ER-000)")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(path=str(DB_PATH))
    try:
        build_index(client, model, load_corpus(CORPUS), args.recreate)
        if args.query:
            for p in search(client, model, args.query, args.k, args.site):
                d = p.payload
                print(f"{p.score:.3f}  {d['doc_id']:<8} {d['language']} {d['year']} "
                      f"{d['source']:<8} rel={d['reliability']}  {d['text']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
