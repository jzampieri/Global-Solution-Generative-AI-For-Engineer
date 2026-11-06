"""Teste de integração da API do Assistente Encosta (contrato POST /assistente/perguntar).

Uso (com a API rodando):  python scripts/testar_api.py http://localhost:8001
Salva as requisições e respostas reais em docs/teste_api.json.
"""

import json
import sys
import time
from pathlib import Path

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
SAIDA = Path(__file__).resolve().parents[1] / "docs" / "teste_api.json"

CASOS = [
    ("Pergunta respondível (RAG)", {"pergunta": "A porta da minha casa começou a emperrar. Isso é sinal de quê?", "modo": "rag"}, 200),
    ("Salvaguarda: sair ou ficar em casa", {"pergunta": "Choveu a noite toda e apareceu uma rachadura nova na parede. Devo sair de casa?", "modo": "rag"}, 200),
    ("Agente com setor (dados da API do produto)", {"pergunta": "Está chovendo muito aqui, o que devo fazer?", "setor_id": "vila_sahy", "modo": "agente"}, 200),
    ("Fora do tema: recusa pelo limiar", {"pergunta": "Qual o melhor time de futebol do Brasil?", "modo": "rag"}, 200),
    ("Do tema, mas não consta na base", {"pergunta": "Quantas pessoas morreram nos deslizamentos de Petrópolis em 2022?", "modo": "rag"}, 200),
    ("Mesmo modelo sem contexto (comparação)", {"pergunta": "No Alerta Máximo, o que a Defesa Civil de Santos orienta as famílias a fazer?", "modo": "sem_contexto"}, 200),
    ("Entrada inválida: pergunta curta demais", {"pergunta": "oi"}, 422),
    ("Entrada inválida: modo inexistente", {"pergunta": "Qual o telefone da Defesa Civil?", "modo": "chatgpt"}, 422),
]

resultados = []
cli = httpx.Client(base_url=URL, timeout=180)

h = cli.get("/health")
print("GET /health ->", h.status_code, h.json())
resultados.append({"caso": "Saúde da API", "metodo": "GET /health", "status": h.status_code, "resposta": h.json()})

for nome, corpo, esperado in CASOS:
    t0 = time.perf_counter()
    r = cli.post("/assistente/perguntar", json=corpo)
    dt = time.perf_counter() - t0
    ok = r.status_code == esperado
    print(f"[{'OK' if ok else 'FALHOU'}] {nome}: {r.status_code} em {dt:.1f}s")
    corpo_resp = r.json()
    if r.status_code == 200:
        print("    ", corpo_resp["resposta"][:150].replace("\n", " "), "| recusa:", corpo_resp["recusa"],
              "| fontes:", [f["doc_id"] for f in corpo_resp["fontes"]])
    resultados.append({"caso": nome, "metodo": "POST /assistente/perguntar", "requisicao": corpo,
                       "status": r.status_code, "esperado": esperado, "ok": ok, "latencia_s": round(dt, 2),
                       "resposta": corpo_resp})

m = cli.get("/metrics")
linhas = [l for l in m.text.splitlines() if l.startswith("http_requests_total") or l.startswith("http_request_duration_seconds_count")]
print("GET /metrics ->", m.status_code, "|", len(linhas), "séries de métricas HTTP")
resultados.append({"caso": "Métricas Prometheus", "metodo": "GET /metrics", "status": m.status_code, "amostra": linhas[:6]})

SAIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
falhas = [r for r in resultados if r.get("ok") is False]
print(f"\n{len(CASOS) - len(falhas)}/{len(CASOS)} casos de POST com o status esperado. Salvo em {SAIDA}")
sys.exit(1 if falhas else 0)
