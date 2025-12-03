# Replicacao-Algoritmo-Genetico
Este repositório contém a implementação e replicação dos experimentos apresentados no artigo "Feature selection algorithm based on optimized genetic algorithm and the application in high-dimensional data processing" (Feng, 2024).

O projeto foca na aplicação de um Algoritmo Genético Otimizado (MGA) para seleção de características em dados biomédicos de alta dimensão (expressão gênica), visando resolver a "Maldição da Dimensionalidade" e melhorar a precisão de diagnósticos computacionais.

Equipe:
Caio Arlen Silva Pinheiro
Carlos Sandro da Costa Gonçalves
Hugo Cézar Pereira Martins

O objetivo central deste trabalho foi validar a eficácia do método MGA (Matrix Genetic Algorithm) proposto pelos autores originais. Diferente de algoritmos genéticos tradicionais (GA), o MGA utiliza uma estratégia de inicialização matricial esparsa e operadores genéticos adaptados para acelerar a convergência.

Algoritmo: MGA (Optimized Matrix Genetic Algorithm)

Classificador Base: K-Nearest Neighbors com K=5 (baseado na análise de sensibilidade do artigo)

Avaliação: Validação Cruzada (Cross-Validation) com 5 dobras

Operadores Genéticos:

Seleção por Torneio.

Cruzamento Uniforme (Uniform Crossover).

Mutação Bit-Flip.


Datasets Utilizados: Dados públicos de expressão gênica (Colon, SRBCT, Brain Tumor, Leukemia).

Pré requisitos: 
Python 3.8(minimo) junto das dependências:
"pip install numpy pandas scikit-learn matplotlib scipy"

E ESTRUTURA DO ARQUIVO PRECISA ESTAR ASSIM:

[pasta]/ 
main.py
mga_algorithm.py
README.md
data(pasta)

[data]/ 
(os datasets)

rode o arquivo usando python main.py


Resumo dos resultados:

Colon: 96.8% de acurácia obtida
Lymphoma: 90.5% de acurácia obtida
SRBCT: 100% de acurácia obtida
Leukemia: 98.5% de acurácia obtida


Artigo Base:

    Feng, Y. (2024). Feature selection algorithm based on optimized genetic algorithm and the application in high-dimensional data processing. PLOS ONE, 19(5), e0303088. DOI: 10.1371/journal.pone.0303088
