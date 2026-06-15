"""ChromaDB-backed vector store with an in-memory fallback.

Stores chunk embeddings plus rich metadata (document id, filename, page,
owner) so retrieval results can be turned directly into citations and
filtered per-user. If ChromaDB is unavailable the store falls back to a
simple in-process cosine index, keeping the pipeline runnable.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.logging_config import get_logger
from app.rag.embeddings import embedding_provider

logger = get_logger("rag.vectorstore")


@dataclass
class RetrievedChunk:
    vector_id: str
    text: str
    score: float  # higher = more similar
    metadata: dict


class VectorStore:
    def __init__(self) -> None:
        self._collection = None
        self._fallback: dict[str, dict] = {}
        self._init_chroma()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("chroma_ready", collection=settings.chroma_collection)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("chroma_fallback", error=str(exc))
            self._collection = None

    # ---- writes ----
    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        embeddings = embedding_provider.embed_documents(texts)
        if self._collection is not None:
            self._collection.add(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
            )
        else:
            for i, vid in enumerate(ids):
                self._fallback[vid] = {
                    "text": texts[i],
                    "metadata": metadatas[i],
                    "embedding": embeddings[i],
                }

    def delete_document(self, document_id: str) -> None:
        if self._collection is not None:
            self._collection.delete(where={"document_id": document_id})
        else:
            self._fallback = {
                k: v
                for k, v in self._fallback.items()
                if v["metadata"].get("document_id") != document_id
            }

    # ---- reads ----
    def query(
        self, query_text: str, *, top_k: int, owner_id: str | None = None
    ) -> list[RetrievedChunk]:
        embedding = embedding_provider.embed_query(query_text)
        where = {"owner_id": owner_id} if owner_id else None

        if self._collection is not None:
            res = self._collection.query(
                query_embeddings=[embedding], n_results=top_k, where=where
            )
            out: list[RetrievedChunk] = []
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            for vid, doc, meta, dist in zip(ids, docs, metas, dists):
                out.append(
                    RetrievedChunk(
                        vector_id=vid,
                        text=doc,
                        score=1.0 - float(dist),  # cosine distance → similarity
                        metadata=meta or {},
                    )
                )
            return out

        return self._fallback_query(embedding, top_k, owner_id)

    def _fallback_query(
        self, embedding: list[float], top_k: int, owner_id: str | None
    ) -> list[RetrievedChunk]:
        def cosine(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b))  # both are L2-normalised

        scored = []
        for vid, rec in self._fallback.items():
            if owner_id and rec["metadata"].get("owner_id") != owner_id:
                continue
            scored.append(
                RetrievedChunk(
                    vector_id=vid,
                    text=rec["text"],
                    score=cosine(embedding, rec["embedding"]),
                    metadata=rec["metadata"],
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        if self._collection is not None:
            return self._collection.count()
        return len(self._fallback)


vector_store = VectorStore()
