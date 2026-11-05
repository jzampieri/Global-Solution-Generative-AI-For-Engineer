"""Item 04: geração fundamentada com validação Pydantic, nova tentativa e recusa controlada.

Três configurações para o placar (item 06):
- sem_contexto: o mesmo LLM, sem trechos
- rag:          RAG simples (busca + limiar + geração)
- agente:       ver agent.py
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from . import prompts
from .config import CONFIG
from .llm import OllamaLLM
from .retriever import Buscador
from .schemas import AVISO_EMERGENCIA, Fonte, RespostaAssistente, recusa_controlada

# Salvaguarda de segurança (erro nº 1: o modelo respondeu "Não saia de casa" diante de rachadura nova,
# quando a cartilha da Defesa Civil diz "saia imediatamente"). Orientar a permanecer no local é a
# falha mais perigosa possível neste produto, então ela é bloqueada por regra, não só pelo prompt.
_PERMANECER = re.compile(
    r"n[aã]o (saia|deixe|abandone|precisa sair)|fique (em casa|no local|dentro)|permane[cç]a (em casa|no local)"
    r"|continue em casa|pode ficar em casa",
    re.IGNORECASE,
)
SAIDA_SEGURA = (
    "Diante de sinais de perigo, como rachaduras, trincas ou inclinação de muros, postes e árvores, "
    "saia imediatamente do local e procure abrigo em um lugar sem perigo de deslizamento. "
    "Depois, informe a Defesa Civil pelo 199."
)


def salvaguarda_permanencia(resposta: RespostaAssistente, textos_recuperados: list[str]) -> tuple[RespostaAssistente, bool]:
    """Se a resposta manda ficar no local e nenhum trecho recuperado diz isso, troca pela orientação oficial.

    Retorna (resposta, acionada). A orientação substituta é a da cartilha da Defesa Civil SP
    (doc_id cartilha_dc_sp), que está na base de conhecimento.
    """
    texto = f"{resposta.resposta} {' '.join(resposta.o_que_fazer)}"
    if resposta.recusa or not _PERMANECER.search(texto):
        return resposta, False
    if any(_PERMANECER.search(t) for t in textos_recuperados):
        return resposta, False  # o próprio documento recuperado orienta permanecer: não é invenção
    return RespostaAssistente(
        resposta=SAIDA_SEGURA,
        o_que_fazer=["Saia imediatamente do local", "Procure abrigo em lugar sem perigo de deslizamento",
                     AVISO_EMERGENCIA],
        fontes=[Fonte(doc_id="cartilha_dc_sp", pagina=1)],
        recusa=False,
        confianca=0.5,
    ), True


def gerar_validado(llm: OllamaLLM, mensagens: list[dict], tentativas_extras: int = 1) -> tuple[RespostaAssistente, dict]:
    """Chama o LLM e valida o JSON. Se falhar, pede correção uma vez; depois, recusa controlada.

    Retorna (resposta, meta) com meta = {json_valido_1a, tentativas, latencia_s, bruto}.
    """
    meta = {"json_valido_1a": False, "tentativas": 0, "latencia_s": 0.0, "bruto": []}
    msgs = list(mensagens)
    for tentativa in range(1 + tentativas_extras):
        saida = llm.chat(msgs, json_mode=True)
        meta["tentativas"] += 1
        meta["latencia_s"] += saida["latencia_s"]
        meta["bruto"].append(saida["content"])
        try:
            resposta = RespostaAssistente.model_validate(json.loads(saida["content"]))
            meta["json_valido_1a"] = tentativa == 0
            return resposta, meta
        except (json.JSONDecodeError, ValidationError) as erro:
            msgs += [
                {"role": "assistant", "content": saida["content"]},
                {"role": "user", "content": prompts.CORRECAO_JSON.format(erro=str(erro)[:500])},
            ]
    return recusa_controlada("Não consegui gerar uma resposta confiável."), meta


def filtrar_citacoes(resposta: RespostaAssistente, doc_ids_validos: set[str]) -> tuple[RespostaAssistente, bool]:
    """Remove fontes inventadas (doc_id que não veio na busca). Retorna (resposta, citacao_correta)."""
    if resposta.recusa:
        return resposta, True
    validas = [f for f in resposta.fontes if f.doc_id in doc_ids_validos]
    correta = len(validas) == len(resposta.fontes) and bool(validas)
    if not validas:
        return recusa_controlada("A resposta gerada não pôde ser ligada a um documento oficial."), False
    resposta.fontes = validas
    return resposta, correta


class Assistente:
    def __init__(self, buscador: Buscador | None = None, llm: OllamaLLM | None = None,
                 limiar: float = CONFIG.limiar_recusa, k: int = CONFIG.top_k):
        self.buscador = buscador
        self.llm = llm or OllamaLLM()
        self.limiar = limiar
        self.k = k

    def sem_contexto(self, pergunta: str) -> tuple[RespostaAssistente, dict]:
        msgs = [
            {"role": "system", "content": prompts.SISTEMA_SEM_CONTEXTO},
            {"role": "user", "content": pergunta},
        ]
        resposta, meta = gerar_validado(self.llm, msgs)
        meta.update(modo="sem_contexto", recuperados=[], citacao_correta=None)
        return resposta, meta

    def rag(self, pergunta: str) -> tuple[RespostaAssistente, dict]:
        resultados, score_max = self.buscador.buscar_com_score(pergunta, k=self.k)
        meta_busca = {
            "modo": "rag",
            "score_max": round(score_max, 4),
            "recuperados": [{"chunk_id": r.trecho.chunk_id, "score": round(r.score, 4)} for r in resultados],
        }
        # Limiar: se nem o trecho mais parecido da base passa do limiar, recusa sem chamar o LLM.
        if not resultados or score_max < self.limiar:
            meta_busca.update(json_valido_1a=True, tentativas=0, latencia_s=0.0, citacao_correta=True,
                              recusa_por_limiar=True)
            return recusa_controlada(), meta_busca

        msgs = [
            {"role": "system", "content": prompts.SISTEMA_RAG},
            {"role": "user", "content": f"TRECHOS:\n{prompts.montar_contexto(resultados)}\n\nPERGUNTA: {pergunta}"},
        ]
        resposta, meta = gerar_validado(self.llm, msgs)
        resposta, citacao_ok = filtrar_citacoes(resposta, {r.trecho.doc_id for r in resultados})
        resposta, salvaguarda = salvaguarda_permanencia(resposta, [r.trecho.texto for r in resultados])
        meta.update(meta_busca, citacao_correta=citacao_ok, recusa_por_limiar=False, salvaguarda=salvaguarda)
        return resposta, meta
