from typing import Any
from pathlib import Path
import numpy as np

from app.core.config import settings


class AppStore:
    def __init__(self):
        # ── Corpus ────────────────────────────────────────────────────────
        self.text: str = ""
        self.filename: str = ""

        # ── Chunking ──────────────────────────────────────────────────────
        self.chunks: list[str] = []
        self.best_chunking_method: str = ""

        # ── Embeddings ────────────────────────────────────────────────────
        self.lsa: np.ndarray | None = None
        self.vec: Any = None   # TfidfVectorizer instance
        self.svd: Any = None   # TruncatedSVD instance
        self.pca: Any = None   # PCA instance

        # ── Query history ─────────────────────────────────────────────────
        self.query_history: list[dict] = []

    def load_corpus(self) -> bool:
        """
        Load the corpus from CORPUS_PATH at startup.
        Returns True if successful, False otherwise.
        """
        path = Path(settings.CORPUS_PATH)
        if not path.exists():
            print(f"[store] WARNING: Corpus not found at '{path}'. "
                  f"Make sure data_final_cleaned.txt is in PFA/data/")
            return False
        try:
            self.text = path.read_text(encoding="utf-8", errors="replace")
            self.filename = path.name
            print(f"[store] Corpus loaded: '{self.filename}' "
                  f"({len(self.text):,} chars)")
            return True
        except Exception as e:
            print(f"[store] ERROR loading corpus: {e}")
            return False

    def reset(self):
        """Clear everything except the corpus text."""
        self.chunks = []
        self.best_chunking_method = ""
        self.lsa = None
        self.vec = None
        self.svd = None
        self.pca = None
        self.query_history = []


# Single instance imported by all routers and services
store = AppStore()