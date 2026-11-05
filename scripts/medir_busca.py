"""Item 03: indexa a base, mede recall@1/@3 (geral e por grupo), escolhe o limiar e gera o gráfico."""
import json
import sys
from collections import defaultdict
from pathlib import Path

raiz = Path(sys.argv[1])
sys.path.insert(0, str(raiz / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assistente.evaluate import carregar_perguntas
from assistente.ingest import carregar
from assistente.retriever import Buscador, melhor_limiar, recall_at_k, scores_top1

buscador = Buscador(carregar()).indexar()
buscador.salvar()
perguntas = carregar_perguntas()

geral = recall_at_k(buscador, perguntas)
por_grupo = {}
for g in sorted({p.grupo for p in perguntas if not p.deve_recusar}):
    por_grupo[g] = recall_at_k(buscador, [p for p in perguntas if p.grupo == g])

# Erros de busca (gabarito fora do top-3), para a análise de erros.
falhas = []
for p in perguntas:
    if p.deve_recusar:
        continue
    top = buscador.buscar(p.pergunta, k=3)
    ids = [r.trecho.chunk_id for r in top]
    if not set(ids) & set(p.chunks_gabarito):
        falhas.append({"id": p.id, "pergunta": p.pergunta, "gabarito": p.chunks_gabarito,
                       "top3": [(r.trecho.chunk_id, round(r.score, 4)) for r in top]})

s = scores_top1(buscador, perguntas)
lim, acc = melhor_limiar(s)
resp = sorted(x["score"] for x in s if not x["deve_recusar"])
nc = sorted(x["score"] for x in s if x["deve_recusar"])

# Erros com o limiar escolhido
recusa_indevida = [x["id"] for x in s if not x["deve_recusar"] and x["score"] < lim]
aceita_indevida = [x["id"] for x in s if x["deve_recusar"] and x["score"] >= lim]

fig, ax = plt.subplots(figsize=(8, 4))
bins = [round(0.70 + i * 0.005, 3) for i in range(0, 41)]
ax.hist(resp, bins=bins, alpha=0.7, label=f"respondível (n={len(resp)})", color="#2a7f62")
ax.hist(nc, bins=bins, alpha=0.7, label=f"não consta (n={len(nc)})", color="#b5532b")
ax.axvline(lim, color="k", ls="--", label=f"limiar = {lim:.3f} (acurácia {acc:.0%})")
ax.set_xlabel("maior similaridade de cosseno entre a pergunta e a base (multilingual-e5-small)")
ax.set_ylabel("nº de perguntas")
ax.set_title("Separação entre perguntas respondíveis e 'não consta'")
ax.legend()
plt.tight_layout()
(raiz / "docs").mkdir(exist_ok=True)
plt.savefig(raiz / "docs" / "limiar.png", dpi=150)

resultado = {
    "modelo": buscador.nome_modelo,
    "n_trechos": len(buscador.trechos),
    "n_respondiveis": len(resp),
    "n_nao_consta": len(nc),
    "recall": geral,
    "recall_por_grupo": por_grupo,
    "limiar": lim,
    "acuracia_limiar": acc,
    "score_respondivel": {"min": resp[0], "mediana": resp[len(resp) // 2], "max": resp[-1]},
    "score_nao_consta": {"min": nc[0], "mediana": nc[len(nc) // 2], "max": nc[-1]},
    "recusa_indevida": recusa_indevida,
    "aceita_indevida": aceita_indevida,
    "falhas_top3": falhas,
}
saida = raiz / "avaliacao" / "resultados" / "busca.json"
saida.parent.mkdir(parents=True, exist_ok=True)
saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in resultado.items() if k != "falhas_top3"}, ensure_ascii=False, indent=1))
print("falhas top3:")
for f in falhas:
    print(" ", f["id"], f["pergunta"][:70], "| gab", f["gabarito"], "| top", f["top3"])
