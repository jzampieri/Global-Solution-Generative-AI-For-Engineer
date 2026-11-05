# Contexto — Generative AI for Engineering (GS 2026.2 · CitizenTech · 4ES)

Fontes: `../Desafio-FIAP-GS-CitizenTech 4ES.pdf` (págs. 28–31) e `../Encosta-Viva-Deslizamentos.pdf` (págs. 5–6).

- **Professor:** Francisco Elanio Bezerra · **Valor:** 10 pts
- **Entrega:** links do repositório no Teams da disciplina até **19/11/2026, 23h55** (um integrante envia)
- **Atividade:** "Treinamento de uma LLM local" — tema energia, **segurança** ou mobilidade
- **Objetivo:** provar que a IA **acerta mais e alucina menos** que o mesmo modelo sem contexto, rodando integrada ao produto de forma segura, explicável e de acordo com a LGPD.
- **Produto da equipe:** Encosta Viva (alerta de deslizamento por setor de risco). Componente desta disciplina: **"Assistente Encosta"**, uma IA local (Ollama + modelo de 1,5B–3B, ex.: Qwen2.5 / Llama 3.2) que responde dúvidas de moradores e agentes, sempre cita o documento oficial e **recusa** quando a resposta não está na base.

## Critérios de avaliação

| # | Item | Pts | O que fazer |
|---|------|-----|-------------|
| 01 | Tarefa + casos de teste com gabarito, incluindo "não consta" | 0,5 | 60–100 perguntas em 4 grupos: sinais de risco, o que fazer, direitos (aluguel social, interdição), níveis de alerta. 15–20% "não consta" |
| 02 | Base de conhecimento em trechos com fonte | 1,0 | Trechos de 300–600 tokens com sobreposição; metadados `doc_id, titulo, pagina, url, licenca` |
| 03 | Busca com recall@1/@3 e limiar justificado | 1,0 | Embeddings multilíngues (`intfloat/multilingual-e5-small` ou `BAAI/bge-m3`) + FAISS/Chroma; histograma de score respondível vs. "não consta" para justificar o limiar |
| 04 | Geração fundamentada: JSON válido, citação e recusa | 2,0 | Validação com Pydantic → 1 nova tentativa com correção → recusa controlada. Prompt de sistema: usar só os trechos, citar `doc_id`, sempre incluir "em emergência ligue 199 ou 193" |
| 05 | Especialização (LoRA **ou** agente) comparada ao RAG simples | 2,0 | A: LoRA com Unsloth no Colab, 300–500 pares revisados. B (recomendada): agente com as ferramentas `consultar_risco(setor)`, `chuva_72h(setor)` e `buscar_base(pergunta)` |
| 06 | Placar das 3 configurações, interpretado | 2,0 | Sem contexto × RAG simples × RAG + LoRA/agente: acerto, alucinação, recusa correta, JSON válido/citação, latência em CPU |
| 07 | 5 erros reais com causa e correção | 1,0 | Ex.: mandou "ficar em casa" quando o doc dizia sair; confundiu níveis do PPDC; fonte errada; outro município; aceitou pergunta fora do tema. Causa: chunk, limiar ou prompt |
| 08 | README e reprodutibilidade | 0,5 | |

## Formato da resposta (contrato `POST /assistente/perguntar`)

```json
{
  "resposta": "Rachaduras novas em paredes e portas que emperram podem indicar movimento do terreno...",
  "o_que_fazer": ["Saia da casa se as rachaduras aumentarem", "Ligue 199 (Defesa Civil)"],
  "fontes": [{"doc_id": "cartilha_dc_sp", "pagina": 4}],
  "recusa": false,
  "confianca": 0.82
}
```

## Base de conhecimento sugerida (usar a versão atual de cada documento)
- Lei 12.608/2012 (PNPDEC): deveres do poder público e direitos do cidadão
- Cartilhas da Defesa Civil de SP e de municípios sobre escorregamentos: sinais de risco, antes/durante/depois
- PPDC de SP: níveis observação/atenção/alerta e chuva acumulada em 72 h
- Materiais do CEMADEN Educação: explicações em linguagem simples
- "Mapeamento de Riscos em Encostas e Margem de Rios" (Ministério das Cidades/IPT): graus R1–R4
- Páginas oficiais de aluguel social e abrigos do município escolhido

## Relatório no GitHub (obrigatório)
1. Nome completo e RM dos integrantes
2. Ficha Técnica com a descrição do que a IA resolve
3. Diagrama do pipeline de IA e dos documentos
4. Tabela da base: fonte, formato, licença, nº de trechos
5. Performance do modelo nas configurações testadas
6. Os 5 erros, a causa e a correção proposta
7. Conclusão técnica + links do repositório e do vídeo

## Estrutura do repositório (pública)
```
notebooks/   com as saídas salvas
base/        trechos e fontes
avaliacao/   teste e placar
README.md    como reproduzir
```
Também: API, conjunto de teste e demonstração de alucinações.

## Regras transversais
- **LGPD:** nenhum dado pessoal no prompt ou no log (só setor, nunca endereço ou telefone). O app avisa que a resposta é orientação geral e não substitui a Defesa Civil.
- **IA responsável:** a IA não decide evacuação; quem decide é a Defesa Civil (humano no circuito).
- **Integração:** o app do cidadão consome o endpoint; o robô (Maker Lab) também chama o assistente para gerar a frase falada; o agente consulta a API de risco e a tabela Ouro (`fato_risco_setor`).
- **Plano da equipe:** base e teste entre 07 e 10/11 · RAG e placar entre 11 e 13/11 · LoRA/agente e os 5 erros entre 14 e 16/11.
