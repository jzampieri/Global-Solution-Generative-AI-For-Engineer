"""Item 02: monta a base de conhecimento em trechos com fonte.

Lê `base/fontes.csv`, extrai o texto dos arquivos em `base/raw/` (PDF, TXT ou MD)
e quebra em trechos de ~300–600 tokens com sobreposição, guardando
doc_id, titulo, pagina, url e licenca de cada trecho em `base/processed/chunks.jsonl`.

Uso:  python -m assistente.ingest
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from .config import CONFIG
from .schemas import Trecho

# Aproximação: 1 token ≈ 0,75 palavra em português.
# 400 palavras ≈ 530 tokens; sobreposição de 60 palavras ≈ 80 tokens.
PALAVRAS_POR_TRECHO = 400
SOBREPOSICAO = 60
MIN_PALAVRAS = 40  # descarta sobras muito curtas (cabeçalho, rodapé)


def _limpar(texto: str) -> str:
    texto = texto.replace("­", "")  # hífen suave
    texto = re.sub(r"(/gid\d+)+", " ", texto)  # glifos sem mapeamento Unicode (fontes embutidas)
    texto = re.sub(r"-\n(\w)", r"\1", texto)  # palavra quebrada no fim da linha
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{2,}", "\n", texto)
    return texto.strip()


def _intervalo(paginas: str | None) -> set[int] | None:
    """'1-5,8' -> {1,2,3,4,5,8}. Vazio -> None (todas as páginas)."""
    if not paginas or not paginas.strip():
        return None
    resultado: set[int] = set()
    for parte in paginas.split(","):
        a, _, b = parte.strip().partition("-")
        resultado.update(range(int(a), int(b or a) + 1))
    return resultado


def extrair_paginas(arquivo: Path, paginas: str | None = None) -> list[tuple[int | None, str]]:
    """Retorna [(pagina, texto)]. Para TXT/MD a página é None.

    `paginas` limita a extração (ex.: '1-5'), útil quando o PDF é um Diário Oficial inteiro
    e só algumas páginas tratam do tema.
    """
    if arquivo.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        filtro = _intervalo(paginas)
        leitor = PdfReader(str(arquivo))
        return [
            (i + 1, _limpar(p.extract_text() or ""))
            for i, p in enumerate(leitor.pages)
            if filtro is None or i + 1 in filtro
        ]
    return [(None, _limpar(arquivo.read_text(encoding="utf-8")))]


def quebrar(texto: str, tamanho: int = PALAVRAS_POR_TRECHO, sobreposicao: int = SOBREPOSICAO) -> list[str]:
    palavras = texto.split()
    if len(palavras) <= tamanho:
        return [" ".join(palavras)] if len(palavras) >= MIN_PALAVRAS else []
    passo = tamanho - sobreposicao
    trechos = []
    for inicio in range(0, len(palavras), passo):
        janela = palavras[inicio : inicio + tamanho]
        if len(janela) < MIN_PALAVRAS:
            break
        trechos.append(" ".join(janela))
        if inicio + tamanho >= len(palavras):
            break
    return trechos


def _recortar(palavras: list[str], paginas: list, ate: str | None, apos: str | None) -> tuple[list[str], list]:
    """Remove ruído do começo e do fim do documento usando marcadores de texto do fontes.csv.

    - remover_ate:  descarta tudo ANTES do marcador (ex.: cabeçalho de assessoria de imprensa).
    - remover_apos: descarta o marcador e tudo DEPOIS (ex.: outro decreto na mesma página do Diário Oficial).
    Motivo: erro nº 1 do relatório (o cabeçalho de imprensa tirava a cartilha do top-5 da busca).
    """
    texto = " ".join(palavras)
    inicio, fim = 0, len(palavras)
    if ate and ate.strip() and ate in texto:
        inicio = len(texto[: texto.index(ate)].split())
    if apos and apos.strip() and apos in texto:
        fim = len(texto[: texto.index(apos)].split())
    return palavras[inicio:fim], paginas[inicio:fim]


def ler_fontes(caminho: Path = CONFIG.fontes_csv) -> list[dict]:
    with caminho.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def construir_base() -> list[Trecho]:
    trechos: list[Trecho] = []
    for fonte in ler_fontes():
        arquivo = CONFIG.dir_raw / fonte["arquivo"]
        if not arquivo.exists():
            print(f"[aviso] arquivo ausente, pulando: {arquivo.name}")
            continue
        # Junta as páginas do documento numa sequência de palavras, lembrando a página de cada uma,
        # para que páginas curtas não gerem trechos abaixo de 300 tokens.
        palavras: list[str] = []
        paginas_por_palavra: list[int | None] = []
        for pagina, texto in extrair_paginas(arquivo, fonte.get("paginas")):
            ws = texto.split()
            palavras += ws
            paginas_por_palavra += [pagina] * len(ws)
        palavras, paginas_por_palavra = _recortar(palavras, paginas_por_palavra,
                                                   fonte.get("remover_ate"), fonte.get("remover_apos"))

        inicio, n = 0, 0
        for pedaco in quebrar(" ".join(palavras)):
            tamanho = len(pedaco.split())
            n += 1
            trechos.append(
                Trecho(
                    chunk_id=f"{fonte['doc_id']}_{n:04d}",
                    doc_id=fonte["doc_id"],
                    titulo=fonte["titulo"],
                    pagina=paginas_por_palavra[inicio],
                    pagina_fim=paginas_por_palavra[inicio + tamanho - 1],
                    url=fonte["url"],
                    licenca=fonte["licenca"],
                    texto=pedaco,
                )
            )
            inicio += PALAVRAS_POR_TRECHO - SOBREPOSICAO
    return trechos


def salvar(trechos: list[Trecho], caminho: Path = CONFIG.chunks_jsonl) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for t in trechos:
            f.write(t.model_dump_json() + "\n")


def carregar(caminho: Path = CONFIG.chunks_jsonl) -> list[Trecho]:
    with caminho.open(encoding="utf-8") as f:
        return [Trecho.model_validate_json(linha) for linha in f if linha.strip()]


def tabela_da_base(trechos: list[Trecho]) -> list[dict]:
    """Tabela exigida no relatório: fonte, formato, licença e número de trechos."""
    contagem = Counter(t.doc_id for t in trechos)
    return [
        {
            "doc_id": f["doc_id"],
            "fonte": f["titulo"],
            "formato": f["formato"],
            "licenca": f["licenca"],
            "trechos": contagem.get(f["doc_id"], 0),
            "url": f["url"],
        }
        for f in ler_fontes()
    ]


if __name__ == "__main__":
    base = construir_base()
    salvar(base)
    print(f"{len(base)} trechos salvos em {CONFIG.chunks_jsonl}")
    for linha in tabela_da_base(base):
        print(json.dumps(linha, ensure_ascii=False))
