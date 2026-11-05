"""Item 05 (opção B): agente com ferramentas integradas ao produto Encosta Viva.

Ferramentas:
- buscar_base(pergunta)     → RAG híbrido na base oficial
- consultar_risco(setor_id) → GET {ENCOSTA_API_URL}/setores/{id}/risco
- chuva_72h(setor_id)       → chuva acumulada (tabela Ouro fato_risco_setor, via API)

Fluxo em duas fases:
1. Planejamento: o LLM recebe as ferramentas, sem exigência de JSON, e decide quais chamar.
   Se não chamar `buscar_base`, ela é executada mesmo assim (salvaguarda; fica registrada como "forcada").
   Motivo: no 1º placar, com o formato JSON no mesmo prompt, o qwen2.5:3b nunca chamava ferramenta.
2. Resposta: os trechos e os dados do setor vão para o mesmo gerador validado do RAG.

Enquanto a API do produto não estiver no ar, as ferramentas de risco usam dados
SIMULADOS (`SIMULADO`), marcados com "simulado": true. Documente isso no relatório.
"""

from __future__ import annotations

import json

import httpx

from . import prompts
from .config import CONFIG
from .rag import Assistente, filtrar_citacoes, gerar_validado, salvaguarda_permanencia
from .schemas import recusa_controlada

SIMULADO = {
    "vila_sahy": {"nivel": "Atenção", "chuva_72h_mm": 95.0, "fatores": ["chuva 72h acima do limiar de atenção"]},
    "petropolis_quitandinha": {"nivel": "Observação", "chuva_72h_mm": 12.0, "fatores": []},
}


def _tool(nome: str, descricao: str, param: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": nome,
            "description": descricao,
            "parameters": {"type": "object", "properties": {param: {"type": "string"}}, "required": [param]},
        },
    }


TOOLS = [
    _tool("buscar_base", "Busca trechos dos documentos oficiais da Defesa Civil sobre deslizamentos.", "pergunta"),
    _tool("consultar_risco", "Nível de risco atual de um setor e os fatores que pesaram.", "setor_id"),
    _tool("chuva_72h", "Chuva acumulada nas últimas 72 horas no setor, em mm.", "setor_id"),
]


def _api_risco(setor_id: str) -> dict:
    try:
        r = httpx.get(f"{CONFIG.encosta_api_url}/setores/{setor_id}/risco", timeout=5)
        r.raise_for_status()
        return {**r.json(), "simulado": False}
    except httpx.HTTPError:
        dado = SIMULADO.get(setor_id)
        return {**dado, "simulado": True} if dado else {"erro": f"setor {setor_id} não encontrado"}


class Agente(Assistente):
    max_passos = 3

    def _buscar(self, consulta: str, usados: dict) -> float:
        resultados, score_max = self.buscador.buscar_com_score(consulta, k=self.k)
        if score_max >= self.limiar:
            for r in resultados:
                usados.setdefault(r.trecho.chunk_id, r)
        return score_max

    def _executar(self, nome: str, args: dict, usados: dict, setor: dict) -> None:
        if nome == "buscar_base":
            self._buscar(args.get("pergunta", ""), usados)
        elif nome in ("consultar_risco", "chuva_72h") and args.get("setor_id"):
            setor[args["setor_id"]] = _api_risco(args["setor_id"])

    def agente(self, pergunta: str, setor_id: str | None = None) -> tuple:
        usuario = pergunta if not setor_id else f"{pergunta}\n(setor do usuário: {setor_id})"
        msgs = [{"role": "system", "content": prompts.SISTEMA_AGENTE_FERRAMENTAS},
                {"role": "user", "content": usuario}]
        usados: dict = {}
        setor: dict = {}
        chamadas: list[dict] = []
        latencia = 0.0

        # Fase 1: planejamento com ferramentas
        for _ in range(self.max_passos):
            saida = self.llm.chat(msgs, json_mode=False, tools=TOOLS)
            latencia += saida["latencia_s"]
            if not saida["tool_calls"]:
                break
            msgs.append({"role": "assistant", "content": saida["content"], "tool_calls": saida["tool_calls"]})
            for call in saida["tool_calls"]:
                nome = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                chamadas.append({"ferramenta": nome, "args": args})
                self._executar(nome, args, usados, setor)
                msgs.append({"role": "tool", "content": "ok"})

        # Salvaguardas: a busca na base oficial sempre acontece; o setor informado sempre é consultado.
        if not any(c["ferramenta"] == "buscar_base" for c in chamadas):
            chamadas.append({"ferramenta": "buscar_base", "args": {"pergunta": pergunta}, "forcada": True})
            self._buscar(pergunta, usados)
        if setor_id and setor_id not in setor:
            chamadas.append({"ferramenta": "consultar_risco", "args": {"setor_id": setor_id}, "forcada": True})
            setor[setor_id] = _api_risco(setor_id)

        meta_agente = {
            "modo": "agente",
            "chamadas": chamadas,
            "recuperados": [{"chunk_id": c, "score": round(r.score, 4)} for c, r in usados.items()],
        }
        setor_valido = {k: v for k, v in setor.items() if "erro" not in v}
        if not usados and not setor_valido:
            meta_agente.update(json_valido_1a=True, tentativas=0, latencia_s=latencia, citacao_correta=True)
            return recusa_controlada(), meta_agente

        # Fase 2: resposta fundamentada
        partes = []
        if usados:
            partes.append("TRECHOS:\n" + prompts.montar_contexto(list(usados.values())[: self.k + 2]))
        if setor_valido:
            partes.append("DADOS DO SETOR (doc_id=api_encosta_viva):\n" + json.dumps(setor_valido, ensure_ascii=False))
        msgs_final = [
            {"role": "system", "content": prompts.SISTEMA_AGENTE},
            {"role": "user", "content": "\n\n".join(partes) + f"\n\nPERGUNTA: {usuario}"},
        ]
        resposta, meta = gerar_validado(self.llm, msgs_final)
        validos = {r.trecho.doc_id for r in usados.values()} | ({"api_encosta_viva"} if setor_valido else set())
        resposta, citacao_ok = filtrar_citacoes(resposta, validos)
        resposta, salvaguarda = salvaguarda_permanencia(resposta, [r.trecho.texto for r in usados.values()])
        meta.update(meta_agente, citacao_correta=citacao_ok, salvaguarda=salvaguarda)
        meta["latencia_s"] += latencia
        return resposta, meta
