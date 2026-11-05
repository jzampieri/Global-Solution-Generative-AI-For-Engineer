"""Configuração central lida de variáveis de ambiente (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv é opcional
    pass

RAIZ = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:3b"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    )
    # TOP_K=5: recall@5 = 0,86 contra 0,72 no top-3 (avaliacao/resultados/busca.json).
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    # Limiar escolhido no notebook 02 (docs/limiar.png).
    limiar_recusa: float = field(default_factory=lambda: float(os.getenv("LIMIAR_RECUSA", "0.84")))
    encosta_api_url: str = field(default_factory=lambda: os.getenv("ENCOSTA_API_URL", "http://localhost:8000"))

    dir_raw: Path = RAIZ / "base" / "raw"
    fontes_csv: Path = RAIZ / "base" / "fontes.csv"
    chunks_jsonl: Path = RAIZ / "base" / "processed" / "chunks.jsonl"
    indice_faiss: Path = RAIZ / "base" / "processed" / "indice.faiss"
    perguntas_jsonl: Path = RAIZ / "avaliacao" / "perguntas.jsonl"
    dir_resultados: Path = RAIZ / "avaliacao" / "resultados"


CONFIG = Config()
