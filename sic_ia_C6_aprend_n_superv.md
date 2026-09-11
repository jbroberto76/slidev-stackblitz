---
theme: default
transition: fade
lineNumbers: true
colorSchema: dark
layout: image
image: /sic.png
backgroundSize: contain
title: Aprendizado Não Supervisionado
description: Inteligência Artificial
exportFilename: sic_ia_c6_aprend_n_superv
author: José Roberto Bezerra
---

<logos-samsung />

---
layout: image
image: /sic.png
backgroundSize: contain
---

<logos-samsung />
<br><br><br><br><br><br><br><br><br><br><br><br><br>
# {{ $slidev.configs.title }}
{{ $slidev.configs.description }}

---

# Objetivos de Aprendizagem

- Introduzir o conceito de Aprendizado Não Supervisionado e seus principais algoritmos

---

# Agenda

- Conceito
- Classificação
- Aplicações
- Exemplos

---
layout: section
---

# Conceito

---
layout: quote
---

> Aprendizado não supervisionado aplica altoritmos de *Machine Learning* para analisar e **agrupar conjuntos de dados não rotulados**

[https://www.ibm.com/br-pt/think/topics/unsupervised-learning](https://www.ibm.com/br-pt/think/topics/unsupervised-learning)

---
layout: quote
---

> Técnica de aprendizado de máquina que possui uma série de variáveis $(X_1, X_2, \dots , X_p )$ que não possuem uma variável objetivo (ou variável resposta Y) no conjunto de dados

---
layout: quote
---

> Como não há uma variável objetivo (ou resposta) associada com as variáveis exploratórias (*features*), o aprendizado não supervisionado procura descobrir um padrão específico ou conhecimento não descoberto para os dados, fazendo uma predição.

---
layout: quote
---

> Unsupervised algorithms are those that experience only *features* but not a supervision signal

---

# Aprendizado Não Supervisionado
Aplicações

- Associação
- *Clustering* (Agrupamento)
- Redução de dimensionalidade

---
layout: quote
---

# Associação

> Método de ANS baseado em regras para encontrar relacionamentos em variáveis de um determinado conjunto de dados

[https://www.ibm.com/br-pt/think/topics/unsupervised-learning](https://www.ibm.com/br-pt/think/topics/unsupervised-learning)

---

# Associação
Aplicação clássica

- Análise de carrinhos de compra de usuários de uma loja *online*
- Ajudam na compreensão de hábitos, comoportamentos
- Relacionam diferentes produtos
- Ajudam a criar ofertas e vendas cruzadas
    - "Clientes que compraram este item também compraram", na Amazon
    - *"Discover Weekly"* <logos-spotify-icon />

---

# Algoritmos a *priori*

- Aplicados para buscas de padrões frequentes, típicos, etc
- A partir conjuntos de itens frequentes (*frequent itemset*), regras de associação são criadas
- **Exemplo**: clientes que compram leite e pão tem alta probabilidade de comprar manteiga

    $\{ leite, pão \} \rarr \{ manteiga \}$

- Aplicam regras do tipo: SE {A} ENTÃO {B}

---

# Algoritmos a *priori*
Busca de *itemsets*

> Se um conjunto de itens é frequente, os subconjuntos também serão frequentes

- O racional acima permite reduzir o espaço de busca aumentando a propabilidade de sugerir um item mais provável para o cliente e evita combinações inúteis
- Os itens podem ser produtos de uma cesta de compras, uma palavra numa pesquisa, código fonte, etc

---

# Algoritmos a *priori*
Geração de regras

> A partir dos *itemsets* frequentes, são geradas regras de associação e feita a medição da relevância utilizando métricas de suporte, confiança e *lift*

- **Suporte**: frequência do item no *dataset*
- **Confiança**: probabilidade condicional $P(Y|X)$
- **Lift**: força da associação em relação ao acaso

---

# Algoritmos a *priori*
Métricas

**Suporte**

Suponha que:
- 60% das compras incluem leite
- 60% das compras incluem pão
- 40% das compras incluem pão e leite

---

# Algoritmos a *priori*
Métricas

**Confiança**

$$
\begin{align*}
Confianca(X \rarr Y) &= \frac{Suporte(X \cap Y)}{Suporte(X)} \\[1em]
Confianca(Leite \rarr Pão) &= \frac{Suporte(Leite \cap Pão)}{Suporte(Leite)} \\[1em]
Confianca(Leite \rarr Pão) &= \frac{0,4}{0,6} \\[1em]
Confianca(Leite \rarr Pão) &= 0,67 \\[1em]
\end{align*}
$$

---

# Algoritmos a *priori*
Métricas

**Lift**

$$
\begin{align*}
Lift(X \rarr Y) &= \frac{Confianca(X \rarr Y)}{Suporte(Y)} \\[1em]
Lift(Leite \rarr Pão) &= \frac{Confianca(Leite \cap Pão)}{Suporte(Pão)} \\[1em]
Lift(Leite \rarr Pão) &= \frac{0,67}{0,6} \\[1em]
Lift(Leite \rarr Pão) &= 1,11 \\[1em]
\end{align*}
$$

---

# Algoritmos a *priori*
Métricas

**Lift**

- $Lift = 1$ Antecedente e consequente são independentes (a presença de X não afeta a chance de Y)
- $Lift > 1$ Associação positiva: X aumenta a chance de Y.
- $Lift < 1$ Associação negativa: X diminui a chance de Y.

---

# Algoritmos a *priori*
Aplicações

- Cesta de compras (*market basket analysis*)
- Detecção de padrões em *logs* para aplicações de Segurança cibernética
- Em Biotecnologia para detectar padrões em genes ou proteínas

---

# Algoritmos a *priori*
Exemplo

```python{*}{class:'!children:text-sm'}
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
dados = pd.DataFrame([
    [1, 1, 0, 0, 1],  # Cliente 1: comprou Leite, Pão, Manteiga
    [1, 0, 1, 1, 0],  # Cliente 2: Leite, Arroz, Feijão
    [0, 1, 1, 0, 0],  # Cliente 3: Pão, Arroz
    [1, 1, 1, 0, 1],  # Cliente 4: Leite, Pão, Arroz, Manteiga
    [0, 0, 1, 1, 0],  # Cliente 5: Arroz, Feijão
], columns=["Leite", "Pão", "Arroz", "Feijão", "Manteiga"])
print("Dataset de transações:")
print(dados)
itemsets_frequentes = apriori(dados, min_support=0.4, use_colnames=True)
print("\nItemsets frequentes:")
print(itemsets_frequentes)
regras = association_rules(itemsets_frequentes, metric="confidence", min_threshold=0.7)
print("\nRegras de associação:")
print(regras[["antecedents", "consequents", "support", "confidence", "lift"]])
```

---
layout: quote
---

# *Clustering*
Agrupamento

> Utilizado para descobrir os atributos de diferentes grupos (*clusters*) fornecendo uma visão dos seus padrões. Não é utilizado para previsões.

---
layout: quote
---

# *Clustering*
Agrupamento

> O objetivo dos algoritmos de *clustering* é agrupar elementos (amostras, indivíduos, etc) com características semelhantes (**alta coesão**) e distinguir claramente (**alta separação**) elementos com características distintas.

---

# *Clustering*
Aplicações

- Reconhecimento de objetos e faces a partir de imagens digitais
- Agrupamento de documentos, músicas ou filmes com diferentes temas
- Segmentação de clientes

---

# *Clustering*
Tipos

- Hierárquico
    - Divisivo (*Top Down*)
    - Aglomerativo (*Bottom Up*)
- Não Hierárquico

---

# *Clustering*

| Algoritmo                         | Ideia principal                                              | Quando usar                                        |
| --------------------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| **K-Means**                       | Divide em *k* grupos minimizando a distância aos centróides. | Dados numéricos, clusters esféricos.               |
| **Hierárquico**                   | Cria árvore de agrupamentos (dendrograma).                   | Exploração inicial, não precisa definir *k* antes. |
| **DBSCAN**                        | Agrupa pontos densos e separa outliers.                      | Clusters de formato irregular, presença de ruído.  |
| **Gaussian Mixture Models (GMM)** | Modela clusters como distribuições normais.                  | Quando os clusters se sobrepõem.                   |
| **Mean-Shift**                    | Encontra áreas de maior densidade.                           | Quando não se sabe o número de clusters.           |

---

# K-Means

- Pertence a uma classe de algoritmos que realiza a classificação posicionando regiões em torno de centróides
- Cada *cluster* é representado por um centróide posicionado na posição média da distância dos membros
- $k$ representa a quantidade de grupos buscados

---

# K-Means
Algoritmo

1. Inicialização
2. Seleção de membros
3. Ajuste na posição dos centróides
4. Reinicialização

[K-means steps](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)

---

# K-Means
Inicialização

- $k$ centróides são aleatoriamente posicionados
- Determina-se o centróide mais próximo de cada ponto segundo a distância Euclidiana
- Outras métricas podem ser aplicadas
    - Distância de *Manhattan*
    - Distância de *Minkowski*

---
layout: image-right
image: https://media.datacamp.com/cms/google/ad_4nxfw85hlpdhvogfbjrxi9jiktdv2memq-leatixmm_u0_jevvnifvpnhfza7b7xxbd3uw_ilm-rqvtf-xe9hxnks01or84mh4hfuy7ejn2dc6j1jzu691cl0f8dfqgdgjogzkp45p0eekkex8rrrxzzbxtol.png
backgroundSize: contain
---

# Distância Euclidiana

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

---

# K-Means
Seleção de membros

- Para cada ponto do *dataset*, verifica-se qual possui o centroíde mais próximo
- Assim, o ponto é agrupado na classe do centroíde mais próximo

[K-means steps](https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/)

---

# K-Means
Incremento

- Calcula-se o SSE (*Sum of Squared Errors*) ou Inércia
- Atualiza-se a posição dos centróides para a média dos pontos do seu *cluster*
- Menores valores de SSE indicam mais compactação do *cluster*

$$
SSE = \sum_{i=1}^k \sum_{x \in C_i} d(x, \mu_i)^2
$$

---

# K-Means
Convergência

> O *k-means* se repete até que os centróides parem de se mover indicando a convergência do algoritmo.

---

# K-Means
Exemplos de código

- [ex_401](https://colab.research.google.com/drive/1OOfWZTw7GL_1MqcvfO9uG35ncu2PnDy_#scrollTo=XrMrSq9TKHIg)
- [ex_402](https://colab.research.google.com/drive/1ONJPdgrX-2q8L1cIs6o5VrdtK_bKRuZd#scrollTo=jtywMTLVWYBu)

---

# K-Means
Quantidade de *clusters*

- O número de *clusters* é uma escolha crucial no K-Means
- O método *Elbow* é bastante aplicado para a determinação de $k$
- Baseia-se no cálculo da inércia para medir o grau de agrupamento

---

# K-Means
Métdodo *Elbow*

[Elbow method](https://colab.research.google.com/drive/1h0Ace7QoVSksOpEX9E_HdWLrbUDZG5xH#scrollTo=QiI95KpvyJNA)

---

# K-Means
Desafios e Limitações

- Escolha da quantidade de *clusters*
- Sensibilidade a inicialização
- Formato e tamanhos similares
- Dificuldade em lidar com *outliers*
- Dificuldade em lidar com dados categóricos

---
layout: quote
---

# DBSCAN
*Density-based Spatial Clustering of Applications with Noise*

> K-means é um método que calcula a média de K-clusters e as distâncias entre cada ponto de dados para o agrupamento. Por outro lado, DBSCAN aplica densidade para criar o mesmo grupo de conjuntos de dados ligados com densidade constante. É um método de agrupamento que é vantajoso para identificar ruídos e *outliers*.

---

# DBSCAN
*Density-based Spatial Clustering of Applications with Noise*

- Sensível a distribuição dos dados
- Afetado pela densidade dos dados em cada *cluster*
- Assume que os dados incluídos no mesmo *cluster* possuem alta densidade

---
image: https://s3.stackabuse.com/media/articles/dbscan-with-scikit-learn-in-python-2.png
layout: image-right
backgroundSize: contain
---

# DBSCAN
Parâmetros

- $\epsilon$, raio de vizinhança ao redor de cada ponto
- `minPts`, a quantidade mínima de pontos em um raio de vizinhança para que a região seja considerada densa

---
image: https://s3.stackabuse.com/media/articles/dbscan-with-scikit-learn-in-python-2.png
layout: image-right
backgroundSize: contain
---

# DBSCAN
Tipos de pontos

- *Core point* (núcleo): tem pelo menos `minPts` dentro do raio $\epsilon$
- *Border point* (borda): não tem `minPts`, mas está dentro da região de um *core point*
- *Noise point* (ruído): não pertence a nenhum *cluster*.

---

# DBSCAN
Passos

- Para cada ponto ainda não visitado
    - Se houver pelo menos `minPts` vizinhos dentro da região, esse ponto vira um *core point* e forma (ou expande) um *cluster*
    - Todos os pontos conectados densamente a ele entram no mesmo *cluster*
    - Pontos que não podem ser atribuídos a nenhum *cluster* são marcados como ruído

---

# DBSCAN
Exemplo de código

[DBSCAN_example](https://colab.research.google.com/drive/1ZI09MRdI4daajSe85dYh2H9oTYpN_fS6#scrollTo=bTBJHh3L5X1Y)

---

# DBSCAN
Vantagens e Desvantagens

- Classifica conjuntos de dados independente da disposição dos dados
- Sensível a densidade dos dados
- Dispensa a necessidade de quantificar os *clusters* previamente
- Conjuntos de dados com diferentes densidades afetam o desempenho

---
layout: quote
---

# Clustering Hierárquico

> Organiza os dados em uma estrutura de árvore (dendrograma), onde cada nível mostra como os pontos (ou grupos de pontos) vão se agrupando ou se dissociando

---

# Clustering Hierárquico

> Diferente do K-Means e do DBSCAN, ele não precisa de centróides ou de um número fixo de *clusters* definido no início. Pode-se "cortar" a árvore em qualquer nível e obter a quantidade de *clusters* desejada

---
layout: quote
---

> Os algoritmos de agrupamento hierárquico utilizam o conceito de matriz de dissimilaridade para decidir quais clusters mesclar ou dividir.

---
layout: quote
---

# Dissimilaridade

> É a distância entre dois pontos de dados, medida por um determinado método de ligação. A dissimilaridade representa:

- A distância Euclidiana (ou outra) entre os pontos
- Um critério de agrupamento segundo as distâncias de pares de pontos

---

# Clustering Hierárquico
Tipos

- Aglomerativo (*bottom-up*)
- Divisivo (*top-down*)

---
image: /linkage.png
layout: image-right
backgroundSize: contain
---

# Clustering Hierárquico
*linkage*, distândia entre *clusters*

- *Single linkage*: distância mínima entre dois pontos, um de cada *cluster*
- *Complete linkage* (vizinho mais distante): distância máxima entre dois pontos

---
image: /linkage.png
layout: image-right
backgroundSize: contain
---

# Clustering Hierárquico
*linkage*, distândia entre *clusters*

- *Average linkage*: distância média entre todos os pontos dos dois *clusters*
- *Ward’s method*: minimiza a variância dentro dos *clusters* (muito usado, parecido com K-Means)

---

# Clustering Hierárquico
Aglomerativo

1. Cada ponto começa como um *cluster*
2. Calcula-se a matriz de distâncias entre todos os clusters
3. Os dois *clusters* mais próximos são unidos segundo a regra de *linkage* escolhida
4. Atualiza-se a matriz de distâncias
5. Repete até restar apenas um *cluster*

---

# Clustering Hierárquico
Código de exemplo

[Dendrograma](https://colab.research.google.com/drive/150NhS2dAhjGTTF_dF9xxFTuOM48YOo-2)

---

# Clustering Hierárquico
Limitações

- Custo computacional alto
- Dificuldade de ser aplicado em grandes *datasets*
- Sensível a ruídos
- Resultado muito sensível a diferentes métodos de *linkage*

---

# Clustering Hierárquico
Aplicações

- Análise de tendências
- Segmentação de clientes
- Comportamento de compra
- Perfil de risco
- Categorizar pacientes em pesquisa clínica
- Agrupamento de caracteres para reconhecimento de texto

---
layout: quote
---

# Análise das Componentes Principais
*Principal Component Analysis*

> A análise de componentes principais, ou PCA, reduz o número de dimensões em grandes *datasets* aos componentes que retêm a maior parte das informações originais. Ela faz isso transformando variáveis potencialmente correlacionadas em um conjunto menor de variáveis, chamadas *componentes principais*

---

# PCA
Conceito

- Os componentes principais são combinações lineares das variáveis originais que têm a variância máxima em comparação com outras combinações lineares
- O máximo de informações é capturada nesses componentes
- Envolve operações de álgebra linear para transformar o conjunto original de dados em um novo sistema de coordenadas
- Os componentes principais representam as direções da variância e máxima dos dados

---

# PCA
Passos

1. Padronização dos dados (normalização ou z-score), pois o PCA é sensível à escala
2. Cálculo da matriz de covariância ou da decomposição SVD.
3. Autovalores e autovetores
    - Autovalores, quantidade de variância contida em cada componente
    - Autovetores, direção (pesos) de cada componente
4. Escolha dos componentes principais: geralmente os primeiros que explicam 70–95% da variância
5. Projeção dos dados nos novos eixos (componentes).

---

# PCA
Vantagens

- Redução da dimensionalidade, logo do custo computacional
- Remoção de variáveis redundantes

---

# PCA
Aplicações

- **Compressão de imagens**, Reduz a dimensionalidade da imagem e retém as informações essenciais. Ela ajuda a criar representações compactas das imagens, tornando-as mais fáceis de armazenar e transmitir
- **Visualização de dados**, Ajuda a visualizar dados de alta dimensão projetando-os em um espaço de menor dimensão, como um gráfico 2D ou 3D. Isso simplifica a interpretação e a exploração de dados

---

# PCA
Código exemplo

[PCA](https://colab.research.google.com/drive/1T5HCnqGhWvCu0Wd0sWnohYyX_t_u8ZLK#scrollTo=6S8TL3Oxdl6p)

---
layout: fact
---

# Dúvidas?

---

# Referências

- [Algoritmo A Priori](https://www.ibm.com/br-pt/think/topics/apriori-algorithm)
- [O que é k-means clustering?](https://www.ibm.com/br-pt/think/topics/k-means-clustering)
- [DBSCAN with Scikit-Learn in Python](https://stackabuse.com/dbscan-with-scikit-learn-in-python/)
- [O que é PCA?](https://www.ibm.com/br-pt/think/topics/principal-component-analysis)
- [Gráfico de dispersão](https://statorials.org/pt/grafico-de-dispersao/)

---
src: /src/end.md
---
