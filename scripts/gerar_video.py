"""Gera o vídeo de demonstração (MP4 1280x720, sem áudio) a partir de resultados reais do projeto.

Fontes do conteúdo:
- docs/teste_api.json        → requisições e respostas reais da API (scripts/testar_api.py)
- avaliacao/placar.csv       → placar final das 3 configurações
- avaliacao/resultados/busca.json e docs/limiar.png → busca e limiar
- base/processed/chunks.jsonl → tamanho da base

Uso:  python scripts/gerar_video.py   (saída em video/demo_assistente_encosta.mp4)
"""

from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "video" / "demo_assistente_encosta.mp4"
W, H, FPS = 1280, 720, 15
FONTES = Path("C:/Windows/Fonts")

# Paleta
FUNDO, TINTA, SUAVE = (250, 248, 244), (28, 33, 31), (96, 104, 100)
ACENTO, VERDE, VERMELHO = (184, 83, 43), (42, 127, 98), (176, 48, 48)
TERM_BG, TERM_TXT, TERM_CHAVE = (24, 28, 27), (226, 230, 226), (232, 160, 110)


def fonte(nome: str, tam: int) -> ImageFont.FreeTypeFont:
    for arq in {"regular": ["segoeui.ttf", "DejaVuSans.ttf"], "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf"],
                "mono": ["consola.ttf", "DejaVuSansMono.ttf"], "monob": ["consolab.ttf", "DejaVuSansMono-Bold.ttf"]}[nome]:
        p = FONTES / arq
        if p.exists():
            return ImageFont.truetype(str(p), tam)
    return ImageFont.load_default()


F = {k: fonte(n, t) for k, (n, t) in {
    "titulo": ("bold", 54), "h1": ("bold", 38), "h2": ("bold", 26), "txt": ("regular", 24),
    "txtb": ("bold", 24), "peq": ("regular", 19), "mono": ("mono", 19), "monob": ("monob", 19),
    "kpi": ("bold", 44), "rodape": ("regular", 16)}.items()}


def novo(secao: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), FUNDO)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=ACENTO)
    if secao:
        d.text((60, 34), secao.upper(), font=F["peq"], fill=ACENTO)
    d.text((60, H - 36), "Assistente Encosta · FIAP Global Solution 2026.2 · Generative AI for Engineering",
           font=F["rodape"], fill=SUAVE)
    return img, d


def escrever(d, xy, texto, f, cor=TINTA, largura=60, espaco=8):
    x, y = xy
    for par in texto.split("\n"):
        for linha in textwrap.wrap(par, largura) or [""]:
            d.text((x, y), linha, font=f, fill=cor)
            y += f.size + espaco
    return y


def bullets(d, xy, itens, largura=62, f=None):
    f = f or F["txt"]
    x, y = xy
    for item in itens:
        d.ellipse([x, y + 11, x + 9, y + 20], fill=ACENTO)
        y = escrever(d, (x + 24, y), item, f, largura=largura) + 10
    return y


def terminal(d, caixa, linhas):
    x0, y0, x1, y1 = caixa
    d.rounded_rectangle(caixa, radius=12, fill=TERM_BG)
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([x0 + 16 + i * 22, y0 + 14, x0 + 28 + i * 22, y0 + 26], fill=c)
    y = y0 + 44
    for texto, estilo in linhas:
        cor = {"cmd": TERM_CHAVE, "ok": (120, 210, 150), "erro": (240, 120, 110), "txt": TERM_TXT,
               "dim": (150, 158, 154)}[estilo]
        f = F["monob"] if estilo == "cmd" else F["mono"]
        for linha in textwrap.wrap(texto, 98) or [""]:
            if y > y1 - 26:
                return
            d.text((x0 + 20, y), linha, font=f, fill=cor)
            y += 26


# ---------------------------------------------------------------- dados reais
api = json.loads((RAIZ / "docs" / "teste_api.json").read_text(encoding="utf-8"))
casos = {c["caso"]: c for c in api}
busca = json.loads((RAIZ / "avaliacao" / "resultados" / "busca.json").read_text(encoding="utf-8"))
placar = {r["modo"]: r for r in csv.DictReader((RAIZ / "avaliacao" / "placar.csv").open(encoding="utf-8"))}
trechos = [json.loads(l) for l in (RAIZ / "base" / "processed" / "chunks.jsonl").open(encoding="utf-8")]
por_doc = Counter(t["doc_id"] for t in trechos)
AVISO = "Orientação geral baseada em documentos oficiais. Não substitui a Defesa Civil."


def limpa(txt: str) -> str:
    return txt.replace(AVISO, "").strip()


def slide_api(secao, titulo, legenda, caso, destaque):
    c = casos[caso]
    img, d = novo(secao)
    d.text((60, 62), titulo, font=F["h1"], fill=TINTA)
    escrever(d, (60, 116), legenda, F["txt"], SUAVE, largura=92)
    r = c["resposta"]
    linhas = [("$ curl -X POST localhost:8001/assistente/perguntar", "cmd"),
              (json.dumps(c["requisicao"], ensure_ascii=False), "txt"), ("", "txt"),
              (f"HTTP {c['status']}  ·  {c['latencia_s']} s", "ok"),
              ("resposta: " + limpa(r["resposta"]), "txt"),
              ("o_que_fazer: " + " | ".join(r["o_que_fazer"]), "dim"),
              ("fontes: " + ", ".join(sorted({f["doc_id"] for f in r["fontes"]})) if r["fontes"] else "fontes: []", "txt"),
              (f"recusa: {str(r['recusa']).lower()}   confianca: {r['confianca']}", "txt")]
    terminal(d, (60, 200, W - 60, 560), linhas)
    cor, msg = destaque
    d.rounded_rectangle([60, 580, W - 60, 650], radius=10, fill=cor)
    escrever(d, (84, 598), msg, F["txtb"], (255, 255, 255), largura=90)
    return img


def f2(v):
    return f"{float(v):.2f}".replace(".", ",")


slides: list[tuple[Image.Image, float]] = []

# 1 · Título
img, d = novo()
d.text((60, 150), "Assistente Encosta", font=F["titulo"], fill=TINTA)
escrever(d, (60, 230), "LLM local que orienta moradores de áreas de risco de deslizamento, sempre citando o documento oficial "
         "e recusando quando a resposta não está na base.", F["txt"], SUAVE, largura=80)
d.text((60, 360), "Componente Inteligência do produto Encosta Viva", font=F["h2"], fill=ACENTO)
escrever(d, (60, 420), "Julio Cesar Zampieri · RM98772\nRicardo Augusto de Matos Filho · RM95906\n"
         "João Gabriel Dias Nascimento · RM99092", F["txt"], TINTA)
slides.append((img, 6))

# 2 · Problema e tarefa
img, d = novo("O problema")
d.text((60, 62), "Quem mora em encosta recebe alerta genérico e não sabe o que fazer", font=F["h1"], fill=TINTA)
bullets(d, (60, 150), [
    "Tarefa da IA: pergunta em linguagem simples → JSON com resposta curta, passos de ação, fontes e campo de recusa.",
    "Modelo local qwen2.5:3b via Ollama: nenhuma pergunta sai da infraestrutura do projeto (LGPD).",
    "Meta: acertar mais e alucinar menos que o mesmo modelo sem contexto.",
    "Humano no circuito: a IA orienta, mas quem decide evacuação é a Defesa Civil.",
], largura=88)
slides.append((img, 8))

# 3 · Pipeline
img, d = novo("Arquitetura")
d.text((60, 62), "Pipeline de IA", font=F["h1"], fill=TINTA)
etapas = [("Pergunta", "sem dado pessoal"), ("Busca híbrida", "e5-small + BM25"), ("Limiar 0,84", "recusa fora do tema"),
          ("LLM local", "só os trechos"), ("Validação", "Pydantic + reparo"), ("Salvaguardas", "citação e segurança")]
x = 60
for i, (a, b) in enumerate(etapas):
    d.rounded_rectangle([x, 200, x + 176, 320], radius=12, fill=(234, 239, 236) if i % 2 == 0 else (244, 232, 224))
    d.text((x + 14, 222), a, font=F["txtb"], fill=TINTA)
    escrever(d, (x + 14, 262), b, F["peq"], SUAVE, largura=18, espaco=4)
    if i < len(etapas) - 1:
        d.polygon([(x + 181, 252), (x + 192, 260), (x + 181, 268)], fill=ACENTO)
    x += 194
bullets(d, (60, 380), [
    "Agente (2 fases): planeja quais ferramentas chamar (buscar_base, consultar_risco, chuva_72h) e depois responde com o mesmo validador.",
    "API FastAPI: POST /assistente/perguntar, consumida pelo app do cidadão e pelo robô sentinela; /metrics para o Grafana.",
], largura=88)
slides.append((img, 9))

# 4 · Base
img, d = novo("Item 02 · Base de conhecimento")
d.text((60, 62), f"{len(trechos)} trechos de documentos oficiais", font=F["h1"], fill=TINTA)
nomes = {"lei_12608_2012": "Lei 12.608/2012 (PNPDEC)", "cemaden_educacao": "CEMADEN Educação (e-book 2023)",
         "ipt_mapeamento_riscos": "Mapeamento de risco R1–R4 (IPT)", "ppdc_santos": "PPDC de Santos (Decreto 7.931/2017)",
         "cartilha_dc_sp": "Cartilha da Defesa Civil SP", "ppdc_sp_2018_2019": "PPDC · Operação Chuvas de Verão"}
y = 150
for doc, n in por_doc.most_common():
    d.text((60, y), nomes.get(doc, doc), font=F["txt"], fill=TINTA)
    d.rectangle([600, y + 6, 600 + n * 18, y + 30], fill=VERDE)
    d.text((612 + n * 18, y), str(n), font=F["txtb"], fill=TINTA)
    y += 52
escrever(d, (60, y + 20), "Trechos de ~380–530 tokens com sobreposição; cada um guarda doc_id, título, página, URL e licença.",
         F["peq"], SUAVE, largura=110)
slides.append((img, 7))

# 5 · Busca e limiar
img, d = novo("Item 03 · Busca e limiar")
d.text((60, 62), "Busca híbrida e limiar de recusa", font=F["h1"], fill=TINTA)
graf = Image.open(RAIZ / "docs" / "limiar.png").convert("RGB")
graf.thumbnail((760, 400))
img.paste(graf, (60, 140))
rec = busca["recall"]
yk = 150
for rot, chave in [("recall@1", "recall@1"), ("recall@3", "recall@3"), ("recall@5", "recall@5")]:
    d.text((870, yk), rot, font=F["txt"], fill=SUAVE)
    d.text((1040, yk - 10), f2(rec[chave]), font=F["kpi"], fill=VERDE)
    yk += 70
escrever(d, (870, yk + 10), f"Limiar {f2(busca['limiar'])}: mantém 96% das perguntas respondíveis e barra as fora do tema.",
         F["peq"], TINTA, largura=34)
slides.append((img, 9))

# 6–10 · Demonstração da API (respostas reais)
slides.append((slide_api("Demonstração · API", "Pergunta respondível",
                         "Resposta curta, passos de ação e fonte oficial. O aviso de emergência vai sempre em o_que_fazer.",
                         "Pergunta respondível (RAG)", (VERDE, "Fundamentada no e-book do CEMADEN, com doc_id citado")), 9))
slides.append((slide_api("Demonstração · Segurança", "Salvaguarda: \"devo sair de casa?\"",
                         "O modelo pequeno chegou a responder \"não saia de casa\" (erro nº 1). Agora a regra troca isso pela orientação oficial.",
                         "Salvaguarda: sair ou ficar em casa", (VERDE, "Orientação da cartilha: sair, procurar abrigo e ligar 199")), 9))
slides.append((slide_api("Demonstração · Integração com o produto", "Agente com setor informado",
                         "O agente chama chuva_72h e consultar_risco (API do Encosta Viva, aqui simulada) e cita os números do setor.",
                         "Agente com setor (dados da API do produto)", (VERDE, "Usa os dados do setor: 95 mm em 72 h, nível Atenção")), 9))
img, d = novo("Demonstração · Recusa em duas camadas")
d.text((60, 62), "Recusa em vez de inventar", font=F["h1"], fill=TINTA)
for i, (caso, rot) in enumerate([("Fora do tema: recusa pelo limiar", "1ª camada · limiar (sem chamar o LLM)"),
                                 ("Do tema, mas não consta na base", "2ª camada · instrução \"use só os trechos\"")]):
    c = casos[caso]
    y0 = 140 + i * 250
    d.text((60, y0), rot, font=F["h2"], fill=ACENTO)
    terminal(d, (60, y0 + 40, W - 60, y0 + 210), [
        ("pergunta: " + c["requisicao"]["pergunta"], "cmd"),
        (f"HTTP {c['status']}  ·  {c['latencia_s']} s  ·  recusa: {str(c['resposta']['recusa']).lower()}", "ok"),
        ("resposta: " + limpa(c["resposta"]["resposta"]), "txt")])
slides.append((img, 9))
slides.append((slide_api("Demonstração · Alucinação", "O mesmo modelo sem contexto",
                         "Sem os trechos, o modelo inventa a orientação e até as fontes. O plano da Defesa Civil manda ir ao ponto de encontro.",
                         "Mesmo modelo sem contexto (comparação)", (VERMELHO, "Inventou \"fique em casa\" e fontes que não existem")), 9))

# 11 · Placar
img, d = novo("Item 06 · Placar")
d.text((60, 62), "90 perguntas · juiz qwen2.5:7b", font=F["h1"], fill=TINTA)
cols = [("Sem contexto", "sem_contexto"), ("RAG simples", "rag"), ("RAG + agente", "agente")]
linhas = [("Acerto", "acerto", True), ("Alucinação", "taxa_alucinacao", False),
          ("Recusa correta \"não consta\"", "recusa_correta_nao_consta", True),
          ("JSON válido (1ª tentativa)", "json_valido_1a", True), ("Latência média (s)", "latencia_media_s", None)]
x0, y0 = 60, 150
for j, (rot, _) in enumerate(cols):
    d.text((520 + j * 240, y0), rot, font=F["txtb"], fill=TINTA)
for i, (rot, chave, maior_melhor) in enumerate(linhas):
    y = y0 + 60 + i * 70
    d.line([x0, y - 14, W - 60, y - 14], fill=(225, 222, 216), width=1)
    d.text((x0, y), rot, font=F["txt"], fill=TINTA)
    vals = [float(placar[m][chave]) for _, m in cols]
    melhor = (max(vals) if maior_melhor else min(vals)) if maior_melhor is not None else None
    for j, v in enumerate(vals):
        txt = f"{v:.1f}".replace(".", ",") if chave == "latencia_media_s" else f2(v)
        d.text((520 + j * 240, y - 6), txt, font=F["kpi"] if v == melhor else F["h2"],
               fill=VERDE if v == melhor else (SUAVE if j == 0 else TINTA))
slides.append((img, 11))

# 12 · Erros
img, d = novo("Item 07 · 5 erros reais")
d.text((60, 62), "Erros reais, causa e correção", font=F["h1"], fill=TINTA)
bullets(d, (60, 140), [
    "1 · Mandou \"não saia de casa\" (cartilha diz \"saia\"). Causa: chunk com cabeçalho e busca. → Corrigido: limpeza, radical no BM25 e salvaguarda.",
    "2 · Deu \"60 mm\" ao Litoral Norte (número de S. J. dos Campos). Causa: limiar e prompt. → Validar o local do número.",
    "3 · R1–R4 \"em 72 horas\" (o certo é 1 ano). Causa: tabela do IPT é imagem. → Transcrever a tabela.",
    "4 · Falou de seguro citando a cartilha. Causa: checagem de citação só por doc_id. → Verificar suporte do trecho.",
    "5 · Agente nunca chamava ferramentas. Causa: prompt exigia JSON junto com tools. → Corrigido: agente em 2 fases.",
], largura=118, f=F["txt"])
slides.append((img, 12))

# 13 · LGPD e conclusão
img, d = novo("IA responsável e conclusão")
d.text((60, 62), "O que fica", font=F["h1"], fill=TINTA)
ac = {m: float(placar[m]["acerto"]) for _, m in cols}
bullets(d, (60, 140), [
    f"Com contexto, o acerto vai de {f2(ac['sem_contexto'])} para {f2(max(ac['rag'], ac['agente']))} e a recusa correta chega a "
    f"{f2(placar['rag']['recusa_correta_nao_consta'])}.",
    "LGPD: LLM local, sem dado pessoal no prompt nem no log (só o setor); toda resposta avisa que não substitui a Defesa Civil.",
    "Segurança por regra, não só por prompt: nenhuma resposta do RAG ou do agente orienta ficar em casa.",
    "Repositório público com notebooks, base, teste, placar e API: github.com/jzampieri/Global-Solution-Generative-AI-For-Engineer",
], largura=90)
slides.append((img, 10))


# ---------------------------------------------------------------- render
def frames():
    anterior = None
    for img, seg in slides:
        arr = np.asarray(img)
        if anterior is not None:
            for k in range(1, 7):  # transição suave de ~0,4 s
                a = k / 7
                yield (anterior * (1 - a) + arr * a).astype(np.uint8)
        for _ in range(int(seg * FPS)):
            yield arr
        anterior = arr


SAIDA.parent.mkdir(exist_ok=True)
with imageio.get_writer(SAIDA, fps=FPS, codec="libx264", quality=8, macro_block_size=8) as w:
    n = 0
    for fr in frames():
        w.append_data(fr)
        n += 1
for i, (img, _) in enumerate(slides, start=1):  # quadros estáticos para conferência
    img.save(SAIDA.parent / f"quadro_{i:02d}.png")
print(f"{SAIDA} · {n / FPS:.0f} s · {len(slides)} cenas")
