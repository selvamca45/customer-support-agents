"""
agents/kb_search.py - the search engine behind the Knowledge Agent.

How it works (a small RAG, "Retrieval-Augmented Generation", pipeline):
1. CHUNK:  every knowledge_base/*.md file is split at its "## " headings, so each
           section (e.g. "Late or delayed orders") becomes one searchable passage.
2. EMBED:  each passage is turned into a vector (a list of numbers that captures
           its MEANING) with OpenAI's embedding model. Vectors are cached in
           kb_index/ so we only pay for them once.
   STORE:  the vectors are stored in a ChromaDB vector database (chroma_db/),
           which does the nearest-neighbour search. If ChromaDB is not installed,
           the same vectors are searched in memory with NumPy instead.
3. SEARCH: the query is embedded too, and we compare it with every passage using
           cosine similarity (meaning match), plus a keyword score (word match).
           This "hybrid" search understands that "parcel not here" means
           "late delivery", and still rewards exact words like "OTP" or "blender".
4. FILTER: passages below a minimum score are dropped. If nothing is left, the
           result is NO_MATCH, so the system admits it does not know instead of
           guessing (this is how test case 6 gets escalated).

Fallbacks, so the app never crashes: no ChromaDB -> in-memory NumPy search;
no embeddings (no network or credit) -> keyword search only.
"""
import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from config import KB_DIR, PROJECT_ROOT, get_api_key

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# Calibrated on our 9 test cases (Step 5): the weakest correct match scored 0.41
# and the strongest wrong match (gift cards) scored 0.25. 0.33 is the midpoint,
# giving an equal safety margin on both sides.
MIN_SCORE = float(os.getenv("KB_MIN_SCORE", "0.33"))          # hybrid search
MIN_KEYWORD_SCORE = float(os.getenv("KB_MIN_KEYWORD_SCORE", "0.34"))  # keyword-only fallback
TOP_K = 4                 # return at most 4 passages
MAX_GAP_FROM_BEST = 0.20  # drop passages much weaker than the best one
CACHE_FILE = PROJECT_ROOT / "kb_index" / "embeddings.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION = "bluepeak_policies"

SEMANTIC_WEIGHT = 0.8
KEYWORD_WEIGHT = 0.2

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "my", "your", "our", "you", "we", "i", "me", "can", "do", "does", "how", "what",
    "when", "where", "why", "not", "no", "yes", "from", "by", "as", "will", "have",
    "has", "had", "get", "got", "please", "want", "need", "help", "customer", "issue",
}


@dataclass
class Chunk:
    """One searchable passage = one '## ' section of one document."""
    source: str    # file name, e.g. "shipping_policy.md"
    title: str     # document title, e.g. "BluePeak Store - Shipping Policy"
    section: str   # section heading, e.g. "Late or delayed orders"
    text: str      # the section's content

    @property
    def embed_text(self) -> str:
        return f"{self.title} - {self.section}\n{self.text}"


def tokenize(text: str) -> set[str]:
    """Lower-case words without stopwords, with a very simple plural stemmer."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set()
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def load_chunks() -> list[Chunk]:
    """Split every markdown file in knowledge_base/ into '## ' sections."""
    chunks: list[Chunk] = []
    for path in sorted(KB_DIR.glob("*.md")):
        title, section, body = path.stem, None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                if section and any(b.strip() for b in body):
                    chunks.append(Chunk(path.name, title, section, "\n".join(body).strip()))
                section, body = line[3:].strip(), []
            elif section:
                body.append(line)
        if section and any(b.strip() for b in body):
            chunks.append(Chunk(path.name, title, section, "\n".join(body).strip()))
    return chunks


class KnowledgeBase:
    def __init__(self) -> None:
        self.chunks = load_chunks()
        self._chunk_terms = [tokenize(c.embed_text) for c in self.chunks]
        self._client = None
        self.vectors: np.ndarray | None = None
        self.collection = None          # ChromaDB collection, when available
        self.store = "keywords only"    # where the vectors live (shown in the app)
        self.mode = "keyword"
        try:
            self.vectors = self._load_or_build_vectors()
            self.store = "in-memory vectors"
            self.mode = "hybrid (embeddings + keywords)"
        except Exception as exc:  # no network / no credit / any API problem
            print(f"[knowledge base] Embeddings unavailable ({type(exc).__name__}: {exc}). "
                  "Using keyword search only.")
            return
        try:
            self.collection = self._sync_chroma()
            self.store = "ChromaDB"
            self.mode = "hybrid (ChromaDB vectors + keywords)"
        except ImportError:
            print("[knowledge base] ChromaDB is not installed; searching the vectors in memory.")
        except Exception as exc:
            print(f"[knowledge base] ChromaDB unavailable ({type(exc).__name__}: {exc}); "
                  "searching the vectors in memory.")

    # ---------- embeddings ----------
    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=get_api_key())
        response = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # unit length

    def _load_or_build_vectors(self) -> np.ndarray:
        cache = {}
        if CACHE_FILE.exists():
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        keys = [hashlib.sha256(f"{EMBED_MODEL}|{c.embed_text}".encode()).hexdigest()
                for c in self.chunks]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            new_vectors = self._embed([self.chunks[i].embed_text for i in missing])
            for i, vec in zip(missing, new_vectors):
                cache[keys[i]] = vec.tolist()
            CACHE_FILE.parent.mkdir(exist_ok=True)
            CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
        self._keys = keys
        return np.array([cache[k] for k in keys], dtype=np.float32)

    # ---------- ChromaDB vector store ----------
    def _sync_chroma(self):
        """Store every passage vector in a persistent ChromaDB collection.
        Passages that changed are added, removed ones are deleted."""
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=str(CHROMA_DIR),
                                           settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection(
            CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        existing = set(collection.get(include=[])["ids"])
        wanted = dict(zip(self._keys, range(len(self.chunks))))
        stale = [i for i in existing if i not in wanted]
        if stale:
            collection.delete(ids=stale)
        new = [k for k in self._keys if k not in existing]
        if new:
            idx = [wanted[k] for k in new]
            collection.add(
                ids=new,
                embeddings=[self.vectors[i].tolist() for i in idx],
                documents=[self.chunks[i].embed_text for i in idx],
                metadatas=[{"source": self.chunks[i].source, "section": self.chunks[i].section}
                           for i in idx],
            )
        return collection

    def _semantic_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query with every passage."""
        if self.collection is None:
            return self.vectors @ query_vector
        # Nearest-neighbour search in ChromaDB. We ask for every passage (the
        # knowledge base is small) with its stored vector, and compute the exact
        # cosine similarity so scores match the calibrated threshold.
        result = self.collection.query(query_embeddings=[query_vector.tolist()],
                                       n_results=len(self.chunks), include=["embeddings"])
        position = {key: i for i, key in enumerate(self._keys)}
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for key, emb in zip(result["ids"][0], result["embeddings"][0]):
            if key in position:
                scores[position[key]] = float(np.dot(np.asarray(emb, dtype=np.float32), query_vector))
        return scores

    # ---------- search ----------
    def keyword_scores(self, query: str) -> np.ndarray:
        """Share of the query's words that appear in each passage (0 to 1)."""
        q = tokenize(query)
        if not q:
            return np.zeros(len(self.chunks))
        return np.array([len(q & terms) / len(q) for terms in self._chunk_terms])

    def score_all(self, query: str) -> list[tuple[Chunk, float]]:
        """Score every passage for the query, best first."""
        scores = self.keyword_scores(query)
        if self.vectors is not None:
            semantic = self._semantic_scores(self._embed([query])[0])  # cosine similarity
            scores = SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * scores
        order = np.argsort(-scores)
        return [(self.chunks[i], float(scores[i])) for i in order]

    @property
    def min_score(self) -> float:
        return MIN_SCORE if self.vectors is not None else MIN_KEYWORD_SCORE

    def search(self, query: str, top_k: int = TOP_K) -> tuple[list[tuple[Chunk, float]], float]:
        """Return (relevant passages, best score). Empty list means NO_MATCH."""
        ranked = self.score_all(query)
        best = ranked[0][1] if ranked else 0.0
        if best < self.min_score:
            return [], best
        results = [(c, s) for c, s in ranked[:top_k]
                   if s >= self.min_score and s >= best - MAX_GAP_FROM_BEST]
        return results, best


_KB: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    """Build the knowledge base once and reuse it (fast after the first call)."""
    global _KB
    if _KB is None:
        _KB = KnowledgeBase()
    return _KB


def rebuild_kb() -> KnowledgeBase:
    """Re-read knowledge_base/ (e.g. after a document was added) and re-index it."""
    global _KB
    _KB = KnowledgeBase()
    return _KB


def format_results(results: list[tuple[Chunk, float]], best: float) -> str:
    """Turn search results into the text the agents read (machine-parsable header)."""
    if not results:
        return (
            "NO_MATCH\n"
            f"BEST_SCORE: {best:.2f}\n"
            "SOURCES: none\n"
            "No BluePeak Store policy document covers this request."
        )
    sources = list(dict.fromkeys(c.source for c, _ in results))
    lines = [
        f"MATCHES: {len(results)}",
        f"BEST_SCORE: {best:.2f}",
        f"SOURCES: {', '.join(sources)}",
        "",
    ]
    for i, (chunk, score) in enumerate(results, 1):
        lines.append(f"[{i}] {chunk.source} > {chunk.section} (score {score:.2f})")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def parse_kb_output(text: str) -> dict:
    """Read the header lines back (used by the log, escalation rules and UI)."""
    info = {"no_match": True, "best_score": 0.0, "sources": []}
    if not isinstance(text, str):
        return info
    info["no_match"] = text.strip().startswith("NO_MATCH") or "SOURCES:" not in text
    m = re.search(r"BEST_SCORE:\s*([0-9.]+)", text)
    if m:
        info["best_score"] = float(m.group(1))
    m = re.search(r"SOURCES:\s*(.+)", text)
    if m and m.group(1).strip().lower() != "none":
        info["sources"] = [s.strip() for s in m.group(1).split(",") if s.strip()]
    return info
