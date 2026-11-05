"""API do Assistente Encosta: contrato POST /assistente/perguntar.

LGPD: o log guarda só modo, setor, latência e se houve recusa; nunca o texto da pergunta
(que pode conter dados pessoais digitados pelo morador).

Uso:  uvicorn assistente.api:app --reload --port 8001
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .config import CONFIG
from .schemas import PerguntaEntrada, RespostaAssistente

log = logging.getLogger("assistente")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

AVISO = "Orientação geral baseada em documentos oficiais. Não substitui a Defesa Civil."
estado: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    from .agent import Agente
    from .ingest import carregar
    from .retriever import Buscador

    buscador = Buscador(carregar())
    buscador.carregar() if CONFIG.indice_faiss.exists() else buscador.indexar()
    estado["agente"] = Agente(buscador)
    yield
    estado.clear()


app = FastAPI(title="Assistente Encosta", version="0.1.0", lifespan=lifespan)

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)  # /metrics para o Grafana (DevOps)
except ImportError:
    pass


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modelo": CONFIG.llm_model, "pronto": "agente" in estado}


@app.post("/assistente/perguntar", response_model=RespostaAssistente)
def perguntar(entrada: PerguntaEntrada) -> RespostaAssistente:
    agente = estado.get("agente")
    if agente is None:
        raise HTTPException(503, "assistente carregando")
    inicio = time.perf_counter()
    if entrada.modo == "sem_contexto":
        resposta, _ = agente.sem_contexto(entrada.pergunta)
    elif entrada.modo == "agente":
        resposta, _ = agente.agente(entrada.pergunta, entrada.setor_id)
    else:
        resposta, _ = agente.rag(entrada.pergunta)
    log.info("modo=%s setor=%s recusa=%s latencia=%.2fs", entrada.modo, entrada.setor_id,
             resposta.recusa, time.perf_counter() - inicio)
    if AVISO not in resposta.resposta:
        resposta.resposta = f"{resposta.resposta}\n\n{AVISO}"
    return resposta
