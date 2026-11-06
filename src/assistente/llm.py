"""Cliente mínimo do Ollama (LLM local). Nenhum dado é enviado para fora da máquina."""

from __future__ import annotations

import time

import httpx

from .config import CONFIG


class OllamaLLM:
    def __init__(self, modelo: str = CONFIG.llm_model, url: str = CONFIG.ollama_url, timeout: float = 180,
                 num_ctx: int = 8192, num_predict: int = 900, repeat_penalty: float = 1.0):
        self.modelo = modelo
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.repeat_penalty = repeat_penalty
        self.url = url.rstrip("/")
        self.timeout = timeout

    def chat(self, mensagens: list[dict], json_mode: bool = True, temperatura: float = 0.1,
             tools: list[dict] | None = None) -> dict:
        """Retorna {'content': str, 'tool_calls': list, 'latencia_s': float}."""
        corpo = {
            "model": self.modelo,
            "messages": mensagens,
            "stream": False,
            # num_ctx: o padrão do Ollama (2048) cortaria 5 trechos de ~530 tokens + instrução.
            # num_predict: teto de tokens gerados; sem ele, o modo JSON de modelos pequenos às vezes
            # gera espaços em branco sem parar (a saída cortada vira JSON inválido → nova tentativa/recusa).
            # repeat_penalty: fica em 1.0 (desligado). Com 1.1 o laço em "fontes" diminuía, mas o modelo deixava
            # de copiar números dos trechos ("de 1 em 1 hora" virou "pluviômetro automático"). O laço é tratado
            # pelo reparo de JSON truncado (rag.carregar_json) e pela nova tentativa com temperatura maior.
            "options": {"temperature": temperatura, "num_ctx": self.num_ctx, "num_predict": self.num_predict,
                        "repeat_penalty": self.repeat_penalty},
        }
        if json_mode and not tools:
            corpo["format"] = "json"
        if tools:
            corpo["tools"] = tools
        inicio = time.perf_counter()
        # Até 3 tentativas com espera crescente para falhas transitórias (ex.: 500 quando o Ollama
        # descarrega um modelo para carregar outro no meio da requisição; timeout de carga).
        for tentativa in range(3):
            try:
                r = httpx.post(f"{self.url}/api/chat", json=corpo, timeout=self.timeout)
                r.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.TransportError):
                if tentativa == 2:
                    raise
                time.sleep(2 * (tentativa + 1))
        msg = r.json()["message"]
        return {
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", []),
            "latencia_s": time.perf_counter() - inicio,
        }
