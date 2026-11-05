# base/raw

Coloque aqui os documentos originais, com o nome de arquivo exato que aparece na coluna `arquivo` de `../fontes.csv`.

- PDFs: baixem da fonte oficial e preencham `url` e `licenca` no CSV.
- Páginas HTML (lei, aluguel social): salvem o texto limpo em `.txt` (UTF-8).
- Usem sempre a versão atual do documento e anotem a data de acesso no relatório.

- Coluna `paginas` (opcional): limita as páginas lidas do PDF, por exemplo `1-5` quando o arquivo é um Diário Oficial inteiro.

**Pendente:** `aluguel_social_municipio.txt`. O site da Prefeitura de São Sebastião bloqueia acesso automático. Abram no navegador a Lei 2.433/2017 (Programa Auxílio Aluguel) e a Lei 3.119/2025, colem o texto nesse arquivo e adicionem a linha correspondente em `fontes.csv`.

Depois, rodem `python -m assistente.ingest`. O comando gera `../processed/chunks.jsonl`.
