"""Contratos de dados: trecho da base, pergunta de teste e resposta do assistente.

A `RespostaAssistente` é o contrato do endpoint POST /assistente/perguntar
consumido pelo app do cidadão e pelo robô sentinela.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AVISO_EMERGENCIA = "Em emergência ligue 199 (Defesa Civil) ou 193 (Bombeiros)."

Grupo = Literal["sinais_de_risco", "o_que_fazer", "direitos", "niveis_de_alerta", "nao_consta"]


class Trecho(BaseModel):
    """Um trecho (chunk) da base de conhecimento."""

    chunk_id: str
    doc_id: str
    titulo: str
    pagina: int | None = None
    pagina_fim: int | None = None
    url: str
    licenca: str
    texto: str


def _so_aviso(texto: str) -> bool:
    """True se o texto é basicamente o aviso 'em emergência ligue 199/193'."""
    t = texto.lower()
    return ("199" in t or "193" in t) and t.lstrip().startswith(("em emergência", "em caso de emergência", "ligue")) \
        and len(t) < 90


class Fonte(BaseModel):
    doc_id: str
    pagina: int | None = None
    trecho: str | None = Field(default=None, description="Trecho curto que sustenta a resposta")


class RespostaAssistente(BaseModel):
    resposta: str
    o_que_fazer: list[str] = Field(default_factory=list)
    fontes: list[Fonte] = Field(default_factory=list)
    recusa: bool = False
    confianca: float = Field(ge=0.0, le=1.0)

    @field_validator("resposta")
    @classmethod
    def _resposta_nao_vazia(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("resposta vazia")
        return v.strip()

    @model_validator(mode="after")
    def _regras(self) -> "RespostaAssistente":
        # Resposta afirmativa precisa citar ao menos uma fonte.
        if not self.recusa and not self.fontes:
            raise ValueError("resposta sem recusa precisa citar ao menos uma fonte")
        # O campo "resposta" não pode ser só o aviso de emergência (erro observado no 1º placar:
        # o modelo pequeno colocava "ligue 199" na resposta em vez de responder à pergunta).
        if not self.recusa and _so_aviso(self.resposta):
            raise ValueError('o campo "resposta" repetiu só o aviso de emergência; responda à pergunta com a '
                             'informação dos trechos e deixe o aviso apenas em "o_que_fazer"')
        # Aviso de emergência sempre presente.
        if not any("199" in passo or "193" in passo for passo in self.o_que_fazer):
            self.o_que_fazer.append(AVISO_EMERGENCIA)
        return self


class PerguntaTeste(BaseModel):
    """Item do conjunto de teste com gabarito."""

    id: str
    grupo: Grupo
    pergunta: str
    resposta_esperada: str | None = Field(default=None, description="None quando grupo == nao_consta")
    chunks_gabarito: list[str] = Field(default_factory=list, description="chunk_ids que contêm a resposta")
    deve_recusar: bool = False

    @model_validator(mode="after")
    def _coerencia(self) -> "PerguntaTeste":
        if self.grupo == "nao_consta" and not self.deve_recusar:
            raise ValueError(f"{self.id}: perguntas 'nao_consta' devem ter deve_recusar=true")
        return self


class PerguntaEntrada(BaseModel):
    """Corpo da requisição POST /assistente/perguntar (sem dados pessoais, LGPD)."""

    pergunta: str = Field(min_length=3, max_length=500)
    setor_id: str | None = Field(default=None, description="Somente o setor; nunca endereço ou telefone")
    modo: Literal["sem_contexto", "rag", "agente"] = "rag"


def recusa_controlada(motivo: str = "A informação não consta na base oficial do assistente.") -> RespostaAssistente:
    return RespostaAssistente(
        resposta=f"{motivo} Procure a Defesa Civil do seu município.",
        o_que_fazer=[AVISO_EMERGENCIA],
        fontes=[],
        recusa=True,
        confianca=0.0,
    )
