# avaliacao

## perguntas.jsonl (item 01)
Uma pergunta por linha, validada por `assistente.schemas.PerguntaTeste`.

| Campo | Regra |
|---|---|
| `grupo` | `sinais_de_risco`, `o_que_fazer`, `direitos`, `niveis_de_alerta` ou `nao_consta` |
| `resposta_esperada` | Gabarito escrito a partir do trecho oficial; `null` quando o grupo é `nao_consta` |
| `chunks_gabarito` | `chunk_id`s de `base/processed/chunks.jsonl` que contêm a resposta (usados no recall@k) |
| `deve_recusar` | `true` nas perguntas "não consta" |

Metas: **60 a 100 perguntas** no total, com **15–20% no grupo `nao_consta`**. Incluam também perguntas fora do tema e perguntas sobre outros municípios.
### Conjunto atual: 90 perguntas
| Grupo | Nº |
|---|---|
| sinais_de_risco | 16 |
| o_que_fazer | 18 |
| direitos | 19 |
| niveis_de_alerta | 21 |
| nao_consta | 16 (17,8%) |

- Os gabaritos foram escritos só com o que está nos trechos citados em `chunks_gabarito`.
- Todo número que aparece numa resposta (199, 40199, 60 mm, 2.200 famílias…) foi conferido no texto do trecho.
- As perguntas "não consta" incluem armadilhas: dados de outro município (Manaus, Porto Alegre), dado em tempo real (nível atual em Ubatuba, previsão do tempo), o limiar atual do Litoral Norte (que não está na base) e perguntas fora do tema.
- **Revisão da equipe:** releiam os gabaritos antes de rodar o placar.
- **Se o texto de aluguel social de São Sebastião entrar na base:** acrescentem perguntas de direitos sobre ele e confiram se nenhuma pergunta "não consta" passou a ter resposta.

## Saídas geradas
- `resultados/respostas_<modo>.csv`: uma linha por pergunta. Revisem a coluna `acerto` e anotem na coluna `revisao_manual`.
- `placar.csv`: o placar das 3 configurações (item 06).
- `erros.md`: os 5 erros reais (item 07).
