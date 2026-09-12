"""Keep the pinned Pyserini install limited to the Lucene BM25 import path."""

from pathlib import Path

import pyserini


def _remove(path: Path, line: str) -> None:
    source = path.read_text()
    if source.count(line) != 1:
        raise RuntimeError(f"unexpected Pyserini 1.0.0 import layout: {path.name}")
    path.write_text(source.replace(line, ""))


root = Path(pyserini.__file__).parent
_remove(
    root / "search" / "lucene" / "__init__.py",
    "from ._impact_searcher import LuceneImpactSearcher, SlimSearcher\n",
)
_remove(
    root / "search" / "lucene" / "__init__.py",
    "from ._hnsw_searcher import LuceneHnswDenseSearcher, LuceneFlatDenseSearcher\n",
)
_remove(
    root / "index" / "lucene" / "__init__.py",
    "from ._indexer import LuceneIndexer, JacksonObjectMapper, JacksonJsonNode\n",
)
