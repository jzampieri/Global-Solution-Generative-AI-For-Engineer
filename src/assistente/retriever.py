"""Item 03: busca híbrida (embeddings multilíngues + BM25, fundidos por RRF) e métricas de recall.

- Denso: `intfloat/multilingual-e5-small` + FAISS. O E5 exige os prefixos "query: " e "passage: ";
  os vetores são normalizados, então o produto interno do FAISS é a similaridade de cosseno.
- Lexical: BM25 sobre título + texto, sem acentos. Recupera termos exatos que o denso perde
  ("embarrigado", "desalojado", "Decreto 7.931").
- Fusão: Reciprocal Rank Fusion (k=60).

A comparação que justificou a escolha está em `avaliacao/resultados/comparacao_busca.json`.

O limiar de recusa usa a MAIOR similaridade de cosseno (denso) entre a pergunta e a base,
porque o score do RRF depende só da posição e não é calibrado.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .config import CONFIG
from .schemas import Trecho

RRF_K = 60
STOPWORDS = set(
    "a o e de da do das dos em no na nos nas um uma uns umas para por com que se ou ao aos à às "
    "é são isso esse essa meu minha seu sua".split()
)


RADICAL = 5  # prefixo usado como radical: "rachadura"/"rachaduras" → "racha", "parede"/"paredes" → "pared"


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sem acento, sem stopwords e reduzido a um radical simples (prefixo de 5 letras).

    O radical resolve plural e flexões ("rachadura" × "rachaduras"), que antes impediam o BM25 de
    achar a cartilha na pergunta da rachadura (erro nº 1). Números ficam inteiros.
    """
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return [w if w.isdigit() else w[:RADICAL]
            for w in re.findall(r"[a-z0-9]+", texto) if w not in STOPWORDS and len(w) > 1]


class BM25:
    def __init__(self, documentos: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenizar(d) for d in documentos]
        self.tf = [Counter(d) for d in self.docs]
        self.df = Counter(w for d in self.docs for w in set(d))
        self.n = len(self.docs)
        self.media = sum(map(len, self.docs)) / max(self.n, 1)
        self.k1, self.b = k1, b

    def scores(self, consulta: str) -> np.ndarray:
        out = np.zeros(self.n)
        for w in tokenizar(consulta):
            if w not in self.df:
                continue
            idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.media)
                    out[i] += idf * f * (self.k1 + 1) / (f + norm)
        return out


def rrf(*ordens: list[int], k: int = RRF_K) -> list[int]:
    """Funde rankings (listas de índices) pela soma de 1/(k + posição)."""
    total: dict[int, float] = {}
    for ordem in ordens:
        for pos, i in enumerate(ordem):
            total[i] = total.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(total, key=total.get, reverse=True)


@dataclass
class Resultado:
    trecho: Trecho
    score: float  # similaridade de cosseno (denso) deste trecho com a pergunta


class Buscador:
    def __init__(self, trechos: list[Trecho], modelo: str = CONFIG.embedding_model, hibrido: bool = True):
        self.trechos = trechos
        self.nome_modelo = modelo
        self.hibrido = hibrido
        self._e5 = "e5" in modelo.lower()
        self.indice = None
        self.bm25 = BM25([f"{t.titulo} {t.texto}" for t in trechos]) if hibrido else None

    @cached_property
    def modelo(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.nome_modelo)

    def _embed(self, textos: list[str], tipo: str) -> np.ndarray:
        if self._e5:
            textos = [f"{tipo}: {t}" for t in textos]
        vetores = self.modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vetores, dtype="float32")

    def indexar(self) -> "Buscador":
        import faiss

        vetores = self._embed([t.texto for t in self.trechos], "passage")
        self.indice = faiss.IndexFlatIP(vetores.shape[1])
        self.indice.add(vetores)
        return self

    def salvar(self, caminho=CONFIG.indice_faiss) -> None:
        import faiss

        faiss.write_index(self.indice, str(caminho))

    def carregar(self, caminho=CONFIG.indice_faiss) -> "Buscador":
        import faiss

        self.indice = faiss.read_index(str(caminho))
        return self

    def _densos(self, pergunta: str) -> tuple[list[int], dict[int, float]]:
        if self.indice is None:
            raise RuntimeError("índice não carregado: chame indexar() ou carregar()")
        scores, ids = self.indice.search(self._embed([pergunta], "query"), len(self.trechos))
        ordem = [int(i) for i in ids[0] if i >= 0]
        return ordem, {int(i): float(s) for s, i in zip(scores[0], ids[0]) if i >= 0}

    def buscar_com_score(self, pergunta: str, k: int = CONFIG.top_k) -> tuple[list[Resultado], float]:
        """Retorna (top-k trechos, maior similaridade de cosseno da pergunta com a base)."""
        ordem_densa, cos = self._densos(pergunta)
        ordem = ordem_densa
        if self.hibrido:
            ordem_bm25 = list(np.argsort(-self.bm25.scores(pergunta)))
            ordem = rrf(ordem_densa, [int(i) for i in ordem_bm25])
        resultados = [Resultado(self.trechos[i], cos[i]) for i in ordem[:k]]
        return resultados, (cos[ordem_densa[0]] if ordem_densa else 0.0)

    def buscar(self, pergunta: str, k: int = CONFIG.top_k) -> list[Resultado]:
        return self.buscar_com_score(pergunta, k)[0]

    def score_max(self, pergunta: str) -> float:
        return self.buscar_com_score(pergunta, 1)[1]


def recall_at_k(buscador: Buscador, perguntas, ks=(1, 3, 5)) -> dict[str, float]:
    """Fração das perguntas respondíveis cujo chunk do gabarito aparece no top-k."""
    respondiveis = [p for p in perguntas if not p.deve_recusar and p.chunks_gabarito]
    acertos = {k: 0 for k in ks}
    for p in respondiveis:
        ids = [r.trecho.chunk_id for r in buscador.buscar(p.pergunta, k=max(ks))]
        for k in ks:
            if set(ids[:k]) & set(p.chunks_gabarito):
                acertos[k] += 1
    n = len(respondiveis) or 1
    return {f"recall@{k}": acertos[k] / n for k in ks}


def scores_top1(buscador: Buscador, perguntas) -> list[dict]:
    """Maior similaridade (denso) por pergunta, para o gráfico de escolha do limiar."""
    return [{"id": p.id, "deve_recusar": p.deve_recusar, "score": buscador.score_max(p.pergunta)} for p in perguntas]


def melhor_limiar(scores: list[dict]) -> tuple[float, float]:
    """Limiar que maximiza a acurácia de separação respondível × 'não consta'.

    Retorna (limiar, acurácia). Abaixo do limiar → recusa. Em empate, fica o menor limiar
    (prefere responder a recusar indevidamente quem tem resposta na base).
    """
    candidatos = sorted({round(s["score"], 3) for s in scores})
    melhor = (0.0, 0.0)
    for lim in candidatos:
        corretos = sum((s["score"] < lim) == s["deve_recusar"] for s in scores)
        acc = corretos / len(scores)
        if acc > melhor[1]:
            melhor = (lim, acc)
    return melhor
