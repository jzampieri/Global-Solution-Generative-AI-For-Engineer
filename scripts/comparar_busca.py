"""Compara configurações de busca (mesmos trechos, mesmo gabarito): modelos densos e híbrido BM25+denso (RRF)."""
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

raiz = Path(sys.argv[1])
sys.path.insert(0, str(raiz / "src"))
import numpy as np
from sentence_transformers import SentenceTransformer

from assistente.evaluate import carregar_perguntas
from assistente.ingest import carregar

trechos = carregar()
perguntas = carregar_perguntas()
resp = [p for p in perguntas if not p.deve_recusar]

STOP = set("a o e de da do das dos em no na nos nas um uma para por com que se ou ao à é são isso meu minha".split())


def tok(t):
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", t) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tok(d) for d in docs]
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        self.df = Counter(w for d in self.docs for w in set(d))
        self.N, self.k1, self.b = len(docs), k1, b
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, q):
        out = np.zeros(self.N)
        for w in tok(q):
            if w not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(self.docs[i]) / self.avg))
        return out


def denso(nome):
    m = SentenceTransformer(nome)
    e5 = "e5" in nome
    D = m.encode([("passage: " if e5 else "") + t.texto for t in trechos], normalize_embeddings=True)
    Q = m.encode([("query: " if e5 else "") + p.pergunta for p in perguntas], normalize_embeddings=True)
    return Q @ D.T  # (n_perguntas, n_trechos)


def rrf(*rankings, k=60):
    tot = np.zeros(len(trechos))
    for r in rankings:
        for pos, i in enumerate(r):
            tot[i] += 1 / (k + pos + 1)
    return np.argsort(-tot)


def recall(ordens):
    idx = {p.id: i for i, p in enumerate(perguntas)}
    out = {}
    for k in (1, 3, 5):
        hits = sum(bool({trechos[j].chunk_id for j in ordens[idx[p.id]][:k]} & set(p.chunks_gabarito)) for p in resp)
        out[f"recall@{k}"] = round(hits / len(resp), 3)
    return out


bm = BM25([t.titulo + " " + t.texto for t in trechos])
S_bm = np.array([bm.scores(p.pergunta) for p in perguntas])
ord_bm = [list(np.argsort(-s)) for s in S_bm]

resultados = {"bm25": recall(ord_bm)}
for nome in sys.argv[2:]:
    t0 = time.time()
    S = denso(nome)
    ord_d = [list(np.argsort(-s)) for s in S]
    ord_h = [list(rrf(a, b)) for a, b in zip(ord_d, ord_bm)]
    resultados[nome] = recall(ord_d)
    resultados[nome + " + BM25 (RRF)"] = recall(ord_h)
    resultados[nome]["tempo_s"] = round(time.time() - t0, 1)
    print(nome, resultados[nome], resultados[nome + " + BM25 (RRF)"], flush=True)

print(json.dumps(resultados, ensure_ascii=False, indent=1))
(raiz / "avaliacao" / "resultados" / "comparacao_busca.json").write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
