"""Testes sem LLM nem modelo de embedding: rodam no CI em segundos."""

import json

import pytest
from pydantic import ValidationError

from assistente.ingest import quebrar
from assistente.rag import filtrar_citacoes, gerar_validado
from assistente.retriever import melhor_limiar
from assistente.schemas import AVISO_EMERGENCIA, PerguntaTeste, RespostaAssistente


def test_resposta_sem_fonte_e_sem_recusa_e_invalida():
    with pytest.raises(ValidationError):
        RespostaAssistente(resposta="Saia de casa.", fontes=[], recusa=False, confianca=0.9)


def test_aviso_emergencia_e_incluido():
    r = RespostaAssistente(resposta="x", fontes=[{"doc_id": "a"}], confianca=0.5)
    assert AVISO_EMERGENCIA in r.o_que_fazer


def test_resposta_so_com_aviso_e_invalida():
    with pytest.raises(ValidationError):
        RespostaAssistente(resposta="Em emergência ligue 199 (Defesa Civil) ou 193 (Bombeiros).",
                           fontes=[{"doc_id": "a"}], confianca=0.9)
    # mas "O telefone da Defesa Civil é 199." é uma resposta válida
    RespostaAssistente(resposta="O telefone da Defesa Civil é 199.", fontes=[{"doc_id": "a"}], confianca=0.9)


def test_nao_consta_exige_recusa():
    with pytest.raises(ValidationError):
        PerguntaTeste(id="q1", grupo="nao_consta", pergunta="?", deve_recusar=False)


def test_quebrar_com_sobreposicao():
    texto = " ".join(f"p{i}" for i in range(1000))
    trechos = quebrar(texto, tamanho=400, sobreposicao=60)
    assert len(trechos) == 3
    assert trechos[0].split()[-60:] == trechos[1].split()[:60]


def test_citacao_inventada_vira_recusa():
    r = RespostaAssistente(resposta="x", fontes=[{"doc_id": "inventado"}], confianca=0.9)
    r2, ok = filtrar_citacoes(r, {"cartilha_dc_sp"})
    assert r2.recusa and not ok


def test_melhor_limiar_separa():
    scores = [{"score": 0.9, "deve_recusar": False}, {"score": 0.88, "deve_recusar": False},
              {"score": 0.70, "deve_recusar": True}, {"score": 0.75, "deve_recusar": True}]
    lim, acc = melhor_limiar(scores)
    assert acc == 1.0 and 0.75 < lim <= 0.88


def test_bm25_acha_termo_exato_sem_acento():
    from assistente.retriever import BM25

    bm = BM25(["muro embarrigado e trinca no terreno", "chuva forte no verão", "lei de defesa civil"])
    assert int(bm.scores("Meu muro está EMBARRIGADO").argmax()) == 0


def test_rrf_premia_consenso():
    from assistente.retriever import rrf

    assert rrf([0, 1, 2], [1, 0, 2])[2] == 2
    assert rrf([2, 1, 0], [1, 2, 0])[0] in (1, 2)


def test_salvaguarda_troca_ficar_em_casa():
    from assistente.rag import salvaguarda_permanencia

    errada = RespostaAssistente(resposta="Não saia de casa.", fontes=[{"doc_id": "cemaden_educacao"}], confianca=0.9)
    r, acionada = salvaguarda_permanencia(errada, ["Os morros são terras altas..."])
    assert acionada and "saia imediatamente" in r.resposta.lower() and r.fontes[0].doc_id == "cartilha_dc_sp"


def test_salvaguarda_nao_mexe_em_resposta_correta_nem_em_recusa():
    from assistente.rag import salvaguarda_permanencia

    certa = RespostaAssistente(resposta="Saia imediatamente e ligue 199.", fontes=[{"doc_id": "a"}], confianca=0.9)
    assert salvaguarda_permanencia(certa, [])[1] is False
    recusa = RespostaAssistente(resposta="Não consta.", recusa=True, confianca=0.0)
    assert salvaguarda_permanencia(recusa, [])[1] is False


def test_tokenizar_reduz_plural():
    from assistente.retriever import tokenizar

    assert tokenizar("rachaduras nas paredes") == tokenizar("rachadura na parede")


def test_repara_json_truncado_em_fontes():
    from assistente.rag import carregar_json

    cortado = ('{"resposta": "Pode ser sinal de deslizamento.", "o_que_fazer": ["Ligue 199"], '
               '"fontes": [{"doc_id": "cemaden_educacao", "pagina": 21}, {"doc_id": "cemaden_educacao", "pag')
    dados, reparado = carregar_json(cortado)
    assert reparado and dados["fontes"] == [{"doc_id": "cemaden_educacao", "pagina": 21}]
    assert dados["recusa"] is False and dados["confianca"] == 0.5
    RespostaAssistente.model_validate(dados)


def test_fontes_duplicadas_sao_removidas():
    r = RespostaAssistente(resposta="x", fontes=[{"doc_id": "a", "pagina": 1}] * 11 + [{"doc_id": "b"}], confianca=0.5)
    r2, ok = filtrar_citacoes(r, {"a", "b"})
    assert ok and [f.doc_id for f in r2.fontes] == ["a", "b"]


class LLMFalso:
    """Primeira saída inválida, segunda válida: testa a nova tentativa."""

    def __init__(self, saidas):
        self.saidas = iter(saidas)

    def chat(self, mensagens, json_mode=True, **_):
        return {"content": next(self.saidas), "tool_calls": [], "latencia_s": 0.01}


def test_retry_corrige_json():
    valido = json.dumps({"resposta": "ok", "fontes": [{"doc_id": "a"}], "recusa": False, "confianca": 0.7})
    resp, meta = gerar_validado(LLMFalso(["não é json", valido]), [])
    assert not resp.recusa and meta["tentativas"] == 2 and not meta["json_valido_1a"]


def test_duas_falhas_viram_recusa_controlada():
    resp, meta = gerar_validado(LLMFalso(["x", "y"]), [])
    assert resp.recusa and meta["tentativas"] == 2
