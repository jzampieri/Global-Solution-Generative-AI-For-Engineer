"""Item 06: placar das três configurações sobre o conjunto de teste.

Definições (registre-as no relatório):
- acerto:            pergunta respondível → não recusou E o juiz LLM considerou a resposta correta
                     frente ao gabarito (prompts.JUIZ); pergunta "não consta" → recusou.
- alucinação:        não recusou E (devia recusar OU não acertou o gabarito), ou seja, afirmou algo
                     que não está sustentado pela base.
- recusa correta:    entre as perguntas "não consta", fração em que recusou.
- JSON válido:       JSON aceito pelo Pydantic já na 1ª tentativa.
- citação correta:   todas as fontes citadas vieram dos trechos recuperados (só RAG/agente).
- latência:          média em segundos (Ollama na GPU RTX 3050; em CPU pura é mais lento).

Por que juiz LLM e não similaridade: no 1º placar (resultados/v1) a similaridade e5 ≥ 0,85
marcou como certas respostas erradas ("a cada 24 horas" quando o gabarito diz "de 1 em 1 hora"),
porque o e5 dá 0,80–0,90 para quase qualquer par do mesmo tema. A similaridade continua no CSV
só como referência. O juiz usa JUIZ_MODEL (padrão qwen2.5:7b, maior que o modelo avaliado,
para reduzir autoviés). A equipe revisa uma amostra (coluna `revisao_manual`).

Uso:  python -m assistente.evaluate --modos sem_contexto rag agente
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import os

from . import prompts
from .config import CONFIG
from .schemas import PerguntaTeste

JUIZ_MODEL = os.getenv("JUIZ_MODEL", "qwen2.5:7b")


def carregar_perguntas(caminho: Path = CONFIG.perguntas_jsonl) -> list[PerguntaTeste]:
    with caminho.open(encoding="utf-8") as f:
        return [PerguntaTeste.model_validate_json(l) for l in f if l.strip()]


def similaridade(buscador, a: str, b: str) -> float:
    va, vb = buscador._embed([a, b], "query")
    return float(np.dot(va, vb))


def julgar(juiz, pergunta: str, gabarito: str, resposta: str) -> tuple[bool, str]:
    """Juiz LLM: a resposta transmite o gabarito sem contradizê-lo? Retorna (correta, motivo)."""
    msgs = [
        {"role": "system", "content": prompts.JUIZ},
        {"role": "user", "content": f"PERGUNTA: {pergunta}\nGABARITO: {gabarito}\nRESPOSTA: {resposta}"},
    ]
    try:
        saida = json.loads(juiz.chat(msgs, json_mode=True, temperatura=0.0)["content"])
        return bool(saida.get("correta")), str(saida.get("motivo", ""))[:200]
    except (json.JSONDecodeError, KeyError, TypeError):
        return False, "juiz devolveu JSON inválido"


def julgar_linhas(linhas: list[dict], perguntas: list[PerguntaTeste], juiz) -> list[dict]:
    """2ª fase: o juiz avalia as respostas já geradas (sem alternar modelos na GPU)."""
    gab = {p.id: p for p in perguntas}
    for n, l in enumerate(linhas, start=1):
        if n % 30 == 0:
            print(f"[juiz] {n}/{len(linhas)}", flush=True)
        p = gab[l["id"]]
        if p.deve_recusar or l["recusa"]:
            continue  # já decidido na 1ª fase
        l["acerto"], l["motivo_juiz"] = julgar(juiz, p.pergunta, p.resposta_esperada or "", l["texto_juiz"])
        l["alucinacao"] = not l["acerto"]
    return linhas


def avaliar(assistente, perguntas: list[PerguntaTeste], modo: str, juiz=None) -> list[dict]:
    """1ª fase: gera as respostas. Com `juiz`, julga na hora; sem, deixa para julgar_linhas()."""
    linhas = []
    for n, p in enumerate(perguntas, start=1):
        if n % 10 == 0:
            print(f"[{modo}] {n}/{len(perguntas)}", flush=True)
        if modo == "sem_contexto":
            resp, meta = assistente.sem_contexto(p.pergunta)
        elif modo == "rag":
            resp, meta = assistente.rag(p.pergunta)
        else:
            resp, meta = assistente.agente(p.pergunta)

        motivo = ""
        texto = resp.resposta + " | " + " ; ".join(resp.o_que_fazer)
        if p.deve_recusar:
            acerto, sim = resp.recusa, None
        else:
            sim = similaridade(assistente.buscador, resp.resposta, p.resposta_esperada or "")
            if resp.recusa:
                acerto, motivo = False, "recusou pergunta respondível"
            elif juiz is not None:
                acerto, motivo = julgar(juiz, p.pergunta, p.resposta_esperada or "", texto)
            else:
                acerto, motivo = False, "pendente"

        linhas.append({
            "id": p.id,
            "grupo": p.grupo,
            "modo": modo,
            "pergunta": p.pergunta,
            "resposta": resp.resposta,
            "fontes": [f.doc_id for f in resp.fontes],
            "recusa": resp.recusa,
            "deve_recusar": p.deve_recusar,
            "similaridade": None if sim is None else round(sim, 4),
            "acerto": acerto,
            "motivo_juiz": motivo,
            "texto_juiz": texto,
            "alucinacao": (not resp.recusa) and (p.deve_recusar or not acerto),
            "json_valido_1a": meta.get("json_valido_1a"),
            "citacao_correta": meta.get("citacao_correta"),
            "latencia_s": round(meta.get("latencia_s", 0.0), 3),
            "recuperados": meta.get("recuperados", []),
            "revisao_manual": "",
        })
    return linhas


def placar(linhas: list[dict]) -> dict:
    n = len(linhas)
    nao_consta = [l for l in linhas if l["deve_recusar"]]
    citaveis = [l for l in linhas if l["citacao_correta"] is not None and not l["recusa"]]
    return {
        "modo": linhas[0]["modo"] if linhas else None,
        "n": n,
        "acerto": sum(l["acerto"] for l in linhas) / n,
        "taxa_alucinacao": sum(l["alucinacao"] for l in linhas) / n,
        "recusa_correta_nao_consta": sum(l["recusa"] for l in nao_consta) / (len(nao_consta) or 1),
        "json_valido_1a": sum(bool(l["json_valido_1a"]) for l in linhas) / n,
        "citacao_correta": (sum(l["citacao_correta"] for l in citaveis) / len(citaveis)) if citaveis else None,
        "latencia_media_s": float(np.mean([l["latencia_s"] for l in linhas])),
    }


def main() -> None:
    import pandas as pd

    from .agent import Agente
    from .ingest import carregar
    from .llm import OllamaLLM
    from .retriever import Buscador

    ap = argparse.ArgumentParser()
    ap.add_argument("--modos", nargs="+", default=["sem_contexto", "rag", "agente"])
    ap.add_argument("--limite", type=int, default=None, help="avaliar só as N primeiras perguntas")
    args = ap.parse_args()

    buscador = Buscador(carregar())
    buscador.carregar() if CONFIG.indice_faiss.exists() else buscador.indexar()
    agente = Agente(buscador)
    perguntas = carregar_perguntas()[: args.limite]
    CONFIG.dir_resultados.mkdir(parents=True, exist_ok=True)

    # Fase 1: gera tudo com o modelo avaliado (salva a cada configuração).
    por_modo = {}
    for modo in args.modos:
        por_modo[modo] = avaliar(agente, perguntas, modo)
        pd.DataFrame(por_modo[modo]).to_csv(CONFIG.dir_resultados / f"respostas_{modo}.csv", index=False)

    # Fase 2: julga tudo com o modelo juiz. Separar as fases evita que os dois modelos disputem
    # a VRAM (no placar v2 a troca de modelo a cada pergunta causou erro 500 no Ollama).
    juiz = OllamaLLM(modelo=JUIZ_MODEL, num_predict=200)
    resumo = []
    for modo, linhas in por_modo.items():
        julgar_linhas(linhas, perguntas, juiz)
        pd.DataFrame(linhas).to_csv(CONFIG.dir_resultados / f"respostas_{modo}.csv", index=False)
        resumo.append(placar(linhas))
        print(json.dumps(resumo[-1], ensure_ascii=False), flush=True)
    pd.DataFrame(resumo).to_csv(CONFIG.dir_resultados.parent / "placar.csv", index=False)


if __name__ == "__main__":
    main()
