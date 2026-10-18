# Fitness | Recomendação de produtos com k-NN

Recomendador que encontra produtos semelhantes em um catálogo de equipamentos de treino.

**Machine learning** · Aplicação Streamlit

[← Portfólio de Thiago Oliveira](https://github.com/thiagoramos20042/thiagoramos20042) · [Catálogo de projetos](https://github.com/thiagoramos20042/thiagoramos20042/blob/main/PROJETOS.md)

## Objetivo

Explorar a descoberta de itens a partir de similaridade entre características dos produtos.

## O que você encontra

- Leitura do catálogo e normalização de atributos.
- Busca por vizinhos próximos com NearestNeighbors.
- Interface com recomendações, imagens e explicação do método.

## Tecnologias

Python · Streamlit · pandas · NumPy · scikit-learn

## Arquivos principais

| Arquivo | Conteúdo |
| --- | --- |
| [recomendacao.py](recomendacao.py) | Aplicação, preparação dos atributos e recomendação. |
| [Strength Training.csv](Strength%20Training.csv) | Catálogo utilizado pelo sistema. |
| [requirements.txt](requirements.txt) | Dependências registradas no ambiente original. |

## Como explorar

Na raiz do repositório, em um ambiente Python separado:

```bash
python -m pip install streamlit pandas numpy scikit-learn
python -m streamlit run recomendacao.py
```

## Contexto do projeto

FitnessGear é o cenário hipotético do projeto. Percentuais de vendas, conversão e lucro apresentados na narrativa original são ilustrativos; não são resultados financeiros medidos.

---

Integra o [portfólio de Thiago Oliveira](https://github.com/thiagoramos20042).
