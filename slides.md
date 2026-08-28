---
theme: default
lineNumbers: true
colorSchema: light
layout: image
image: /sic.png
backgroundSize: contain
title: Aprendizado Supervisionado
description: Inteligência Artificial
exportFilename: sic_ia_c5_aprend_superv
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
Uma Jornada da Regressão à Classificação

---
layout: quote
---

# Objetivos de Aprendizagem

  - Dominar a formulação matemática e os objetivos das tarefas de Regressão e Classificação
  - Aprender a avaliar modelos supervisionados usando métricas adequadas para cada tarefa
  - Entender a intuição, os conceitos matemáticos e a aplicação de algoritmos clássicos
  - Explorar o poder dos métodos Ensemble para melhorar a performance preditiva

---
layout: quote
---

# Agenda

1.  Introdução ao Aprendizado Supervisionado
2.  Regressão: Previsão de Valores Contínuos
3.  Métricas de Avaliação para Regressão
4.  Classificação: Previsão de Rótulos Categóricos
5.  Métricas de Avaliação para Classificação
6.  Algoritmos de Classificação: `Árvore de Decisão`, `Naive Bayes`, `K-Nearest Neighbors (KNN)`, `Support Vector Machine (SVM)`
7.  Métodos Ensemble
8.  Conclusão e Próximos Passos

---
layout: section
---

# 1\. Introdução ao Aprendizado Supervisionado

---
layout: image
image: /c5_1.png
backgroundSize: contain
---

---
layout: quote
---

# O que é Aprendizado Supervisionado?

> O Aprendizado Supervisionado é um paradigma de Aprendizado de Máquina onde o objetivo é aprender uma função que mapeia um vetor de entrada $X$ para uma variável de saída $y$, com base em um conjunto de pares de exemplos de entrada-saída.

---
layout: quote
---

# O que é Aprendizado Supervisionado?

> Em essência, o algoritmo aprende a partir de um conjunto de dados **rotulado** por um "supervisor" (humanos ou processos).

---
layout: quote
---

# Aprendizado Supervisionado 
Princípio Fundamental

- **Entrada**: Um conjunto de dados de treinamento $D = {(X_1, y_1), (X_2, y_2), ..., (X_N, y_N)}$
- **$X_i$**: Vetor de características (*features*) da $i$-ésima amostra
- **$y_i$**: O rótulo (*label*) ou resposta correta para a $i$-ésima amostra

---
layout: quote
---

# Aprendizado Supervisionado 
Princípio Fundamental

- **Objetivo**: Aprender uma função de mapeamento (hipótese) $h: X\rightarrow Y$ que generalize bem para dados nunca vistos.

---
layout: quote
---

> O modelo "treina" ao observar os exemplos e ajusta seus parâmetros internos para minimizar os erros entre suas previsões $\hat{y}$ e os rótulos verdadeiros $y$.

---
layout: image
image: /c5_2.png
backgroundSize: contain
---

---
layout: quote
---

# Divisão Treinamento-Teste

> Para avaliar a capacidade de generalização de um modelo, **NUNCA** o avaliamos com os mesmos dados que usamos para treiná-lo.

1.  **Conjunto de Treinamento**: Usado para que o modelo aprenda os padrões nos dados
2.  **Conjunto de Teste**: Usado para avaliar a performance do modelo treinado em dados "novos" e não vistos

---

# Divisão Treino-Teste

```python{*}{class:'!children:text-lg'}
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

  - `test_size`: Proporção dos dados para o conjunto de teste.
  - `random_state`: Garante a reprodutibilidade da divisão.
  - `stratify=y`: Mantém a mesma proporção de classes nos conjuntos de treino e teste (crucial para dados desbalanceados).

---
layout: quote
---

# Aprendizado Supervisionado
Classificação

> O tipo da variável de saída $y$ define o tipo de tarefa a ser realizado

---

# Aprendizado Supervisionado
Classificação

> 1.  **Regressão** A variável de saída $y$ é um valor **contínuo** e real
- Exemplos
  - Prever o preço de um imóvel
  - Prever a temperatura amanhã
  - Prever o valor de uma ação no futuro

---

# Aprendizado Supervisionado
Classificação

> 2.  **Classificação** A variável de saída $y$ é um valor **categórico** (um rótulo de um conjunto finito de classes)

- Exemplos
  - Classificar um e-mail como "spam" ou "não spam"
  - Diagnosticar se um tumor é "benigno" ou "maligno"

---
layout: section
---

# 2\. Regressão: Previsão de Valores Contínuos

---
layout: quote
---

# Regressão

> O objetivo da regressão é modelar a relação entre um conjunto de variáveis de entrada (independentes) e uma variável de saída contínua (dependente) para fazer previsões numéricas.

---

> O modelo básico é a Regressão Linear

![Linear Regression](/c5_3.png)

---
layout: quote
---

# Regressão Linear Simples

> Modela a relação entre uma única variável de entrada ($x$) e a saída ($y$) como uma linha reta. O objetivo é encontrar os melhores valores para $\beta_0$ e $\beta_1$.

---

# Regressão Linear Simples
$y = \beta_0 + \beta_1 x + \epsilon$

  - $y$: Variável dependente (o que queremos prever).
  - $x$: Variável independente (a característica).
  - $\beta_0$: Intercepto (o valor de $y$ quando $x=0$).
  - $\beta_1$: Coeficiente angular (a inclinação da linha; o quanto $y$ muda para cada unidade de mudança em $x$).
  - $\epsilon$: O termo de erro (a diferença entre o valor real e o previsto).

---
layout: quote
---

# Regressão Linear Múltipla

> Uma extensão para múltiplas variáveis de entrada ($x_1, x_2, ..., x_p$).

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon$$
Ou em notação vetorial:
$$y = X\beta + \epsilon$$

> O objetivo é encontrar o vetor de coeficientes $\beta$ que melhor se ajusta aos dados.

---
layout: quote
---

# Como o Modelo "Aprende"?
Função de Custo

> O modelo aprende minimizando uma **Função de Custo** (ou *Loss Function*). Para a Regressão Linear, a mais comum é o **Erro Quadrático Médio (MSE)**. O objetivo é encontrar os parâmetros $\beta$ que minimizam $J(\beta)$.

---

# Como o Modelo "Aprende"?
Função de Custo

> $J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \frac{1}{2m} \sum_{i=1}^{m} (y_i - h_\beta(X_i))^2$

  - O $\frac{1}{2}$ é uma constante de conveniência para simplificar a derivada durante o gradiente descendente
  - $m$: Número de amostras de treinamento.
  - $\hat{y}_i$: O valor previsto pelo modelo para a amostra $i$.
  - $y_i$: O valor real da amostra $i$.
  - $h_\beta(X_i)$: A hipótese (previsão) do modelo.

---
layout: quote
---

# Minimizando o Custo

> O Gradiente Descendente é um algoritmo de otimização iterativo usado para encontrar o mínimo de uma função

1.  Começa com valores aleatórios para os parâmetros $\beta$.
2.  Calcula o gradiente (a derivada) da função de custo. O gradiente aponta na direção do maior aumento da função.
3.  Atualiza os parâmetros na direção **oposta** ao gradiente.
4.  Repete até a convergência (quando os parâmetros mudam muito pouco).

**Regra de atualização:** $\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} J(\beta)$

- $\alpha$ é a taxa de aprendizado

---
layout: image
image: /c5_4.gif
backgroundSize: contain
---

---
layout: section
---

# 3\. Métricas de Avaliação para Regressão

---
layout: quote
---

# Como Saber se o Modelo de Regressão é Bom?

> Métricas de avaliação quantificam a performance do modelo, comparando os valores previstos com os valores reais no conjunto de teste.

> Não podemos confiar apenas na intuição visual.

---
layout: quote
---

# Erro Absoluto Médio (MAE)

> Mede a média das diferenças absolutas entre as previsões e os valores reais. É fácil de interpretar, pois está na mesma unidade da variável de saída.

$$MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$$

  - **Prós**: Robusto contra *outliers* (valores extremos)
  - **Contras**: Não penaliza erros grandes de forma mais significativa

---
layout: quote
---

# Erro Quadrático Médio (MSE)

> Mede a média dos erros ao quadrado.

$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

  - **Prós**: Penaliza erros grandes de forma exponencial, o que é desejável em muitos casos. É diferenciável, facilitando a otimização.
  - **Contras**: A unidade é o quadrado da unidade original, dificultando a interpretação.

---
layout: quote
---

# Raiz do Erro Quadrático Médio (RMSE)

> É simplesmente a raiz quadrada do MSE.

$$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2}$$

  - **Prós**: Retorna à unidade original da variável de saída, combinando a interpretabilidade do MAE com a sensibilidade a grandes erros do MSE. É a métrica de erro mais comum para regressão.

---
layout: quote
---

# Coeficiente de Determinação
$R^2$

> Mede a proporção da variância na variável dependente que é previsível a partir das variáveis independentes.

$$R^2 = 1 - \frac{\sum_{i=1}^{m} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{m} (y_i - \bar{y})^2}$$

- $\bar{y}$ é a média dos valores reais.

---
layout: quote
---

# Coeficiente de Determinação
Interpretação ($R^2$)

- $R^2$ típico varia entre $0$ e 1.
- $R^2 \approx 1$: O modelo explica 100% da variabilidade dos dados (ajuste perfeito).
- $R^2 \approx 0$: O modelo não é melhor do que simplesmente prever a média dos dados.
<!-- - $R^2 < 0$: O modelo é pior do que a média. -->

---
layout: quote
---

# Exemplo Prático
Regressão com `scikit-learn`

```python {*}{class:'!children:text-xs'}
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Gerar dados sintéticos
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# 2. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 3. Criar e treinar o modelo
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
# 4. Fazer previsões
y_pred = lin_reg.predict(X_test)
# 5. Avaliar o modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
```

---
layout: section
---

# 4\. Classificação: Previsão de Rótulos Categóricos

---
layout: quote
---

# Classificação

> O objetivo da classificação é atribuir um rótulo de classe a uma amostra de entrada com base em suas características. O modelo aprende um **limite de decisão** (*decision boundary*) que separa as diferentes classes no espaço de características.

---
layout: two-cols-header
---

:: left ::

# Limite de Decisão

> É a superfície (linha, plano, hiperplano) que divide o espaço de características em regiões, uma para cada classe. O trabalho do algoritmo de classificação é encontrar o melhor limite de decisão.

:: right ::

![decision](/c5_5.png)

---
layout: image
image: /c5_6.png
backgroundSize: contain
---

---
layout: image
image: /c5_7.png
backgroundSize: contain
---

---

```python {*}{class:'!children:text-xs'}
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Carregando dados
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Testando diferentes seeds (divisões aleatórias)
for seed in [1, 42, 100, 500]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    print(f"Seed {seed}: R² = {score:.4f}")
```

<v-click>

Saída

```bash{*}{lines: false}
Seed 1: R² = 0.5966
Seed 42: R² = 0.5758
Seed 100: R² = 0.6223
Seed 500: R² = 0.6070
```

</v-click>

---

[Exemplo de k-Fold](https://colab.research.google.com/drive/19is_5GEwtq8ooT1MR25nV2H5MUe2GBsN?usp=sharing)

---
layout: quote
---

# O Dilema Viés vs. Variância
*Bias-Variance Tradeoff*

  - **Viés (Bias)**: O erro devido a suposições simplistas no modelo. Um alto viés leva ao **subajuste (*underfitting*)**, onde o modelo é muito simples e não captura a complexidade dos dados.
  - **Variância (Variance)**: O erro devido à sensibilidade do modelo a pequenas flutuações nos dados de treino. Alta variância leva ao **sobreajuste (*overfitting*)**, onde o modelo decora o ruído dos dados de treino e não generaliza bem para novos dados.

---
layout: quote
---

# O Dilema Viés vs. Variância
*Bias-Variance Tradeoff*

> O objetivo é encontrar um equilíbrio, um modelo com complexidade "na medida certa".


---
layout: image
image: /c5_8.png
backgroundSize: contain
---

---
layout: quote
---

# Viés (Bias) vs. Variância

Encontrando o "Sweet Spot"

<img src="/c5_9.png" class="w-3/4 mx-auto" />

---
layout: section
---

# 5\. Métricas de Avaliação para Classificação

---
layout: quote
---

# Como Saber se o Modelo de Classificação é Bom?

> Para classificação, o erro não é contínuo. Precisamos de métricas que contem o número de "acertos" e "erros" de diferentes maneiras, especialmente quando as classes não são balanceadas.

---
layout: quote
---

# Matriz de Confusão

> Uma tabela que resume a performance de um modelo de classificação em problemas binários (Positivo/Negativo, Comprar/Vender, etc)

---

# Matriz de Confusão

<div class="overflow-x-auto">
<table class="table-auto mx-auto text-center border-collapse border border-gray-300">
  <thead>
    <tr>
      <th class="border border-gray-300 p-3 bg-gray-100"></th>
      <th class="color-black text-center border border-gray-300 p-3 bg-gray-100 font-bold">Previsto: Positivo</th>
      <th class="color-black text-center border border-gray-300 p-3 bg-gray-100 font-bold">Previsto: Negativo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Positivo</td>
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-800">Verdadeiro Positivo (TP)</div>        
        <div class="text-xs text-green-600">✅ Acerto</div>
      </td>
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-800">Falso Negativo (FN)</div>
        <div class="text-xs text-red-600">❌ Erro Tipo II</div>
      </td>
    </tr>
    <tr>
      <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Negativo</td>
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-800">Falso Positivo (FP)</div>
        <div class="text-xs text-red-600">❌ Erro Tipo I</div>
      </td>
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-800">Verdadeiro Negativo (TN)</div>
        <div class="text-xs text-green-600">✅ Acerto</div>
      </td>
    </tr>
  </tbody>
</table>
</div>

- **TP**: Previsto positivo, e era positivo. (Acerto)
- **TN**: Previsto negativo, e era negativo. (Acerto)
- **FP (Erro Tipo I)**: Previsto positivo, mas era negativo.
- **FN (Erro Tipo II)**: Previsto negativo, mas era positivo.

---
layout: quote
---

# Acurácia

> A métrica mais simples: a proporção de previsões corretas.

---

$$Acurácia = \frac{TP + TN}{TP + TN + FP + FN}$$

<br>

<div class="overflow-x-auto">
<table class="table-auto mx-auto text-center border-collapse border border-gray-300">
  <tbody>
    <tr>
      <!-- <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Positivo</td> -->
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-800">Verdadeiro Positivo (TP)</div>        
        <div class="text-xs text-green-600"></div>
      </td>
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-100">Falso Negativo (FN)</div>
        <div class="text-xs text-red-600"></div>
      </td>
    </tr>
    <tr>
      <!-- <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Negativo</td> -->
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-100">Falso Positivo (FP)</div>
        <div class="text-xs text-red-600"></div>
      </td>
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-800">Verdadeiro Negativo (TN)</div>
        <div class="text-xs text-green-600"></div>
      </td>
    </tr>
  </tbody>
</table>
</div>




---
layout: quote
---

# Acurácia
Observação

> A acurácia pode ser enganosa em **conjuntos de dados desbalanceados**. Por exemplo, detecção de fraude (99% não-fraude, 1% fraude). Um modelo que sempre prevê "não-fraude" terá 99% de acurácia, mas é inútil.

---
layout: quote
---

# Precisão

> Das vezes que o modelo previu "Positivo", quantas ele acertou? Foco nos Falsos Positivos.

---

$$Precisão = \frac{TP}{TP + FP}$$

<br>

<div class="overflow-x-auto">
<table class="table-auto mx-auto text-center border-collapse border border-gray-300">
  <tbody>
    <tr>
      <!-- <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Positivo</td> -->
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-800">Verdadeiro Positivo (TP)</div>        
        <div class="text-xs text-green-600"></div>
      </td>
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-100">Falso Negativo (FN)</div>
        <div class="text-xs text-red-600"></div>
      </td>
    </tr>
    <tr>
      <!-- <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Negativo</td> -->
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-600">Falso Positivo (FP)</div>
        <div class="text-xs text-red-600"></div>
      </td>
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-100">Verdadeiro Negativo (TN)</div>
        <div class="text-xs text-green-600"></div>
      </td>
    </tr>
  </tbody>
</table>
</div>

---
layout: quote
---

# Precisão
Observação

> Alta precisão é importante quando o custo de um Falso Positivo é alto. Por exemplo, classificação de e-mails. É melhor deixar um spam passar (FN) do que marcar um e-mail importante como spam (FP).

---
layout: quote
---

# Revocação
Recall ou Sensibilidade

> De todos os casos que eram realmente "Positivos", quantos o modelo conseguiu identificar? Foco nos Falsos Negativos.

---

$$\text{Revocação} = \frac{TP}{TP + FN}$$

<br>

<div class="overflow-x-auto">
<table class="table-auto mx-auto text-center border-collapse border border-gray-300">
  <tbody>
    <tr>
      <!-- <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Positivo</td> -->
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-800">Verdadeiro Positivo (TP)</div>        
        <div class="text-xs text-green-600"></div>
      </td>
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-600">Falso Negativo (FN)</div>
        <div class="text-xs text-red-600"></div>
      </td>
    </tr>
    <tr>
      <!-- <td class="color-black border border-gray-300 p-3 bg-gray-100 font-bold">Real: Negativo</td> -->
      <td class="border border-gray-300 p-3 bg-red-200">
        <div class="font-bold text-red-100">Falso Positivo (FP)</div>
        <div class="text-xs text-red-600"></div>
      </td>
      <td class="border border-gray-300 p-3 bg-green-200">
        <div class="font-bold text-green-100">Verdadeiro Negativo (TN)</div>
        <div class="text-xs text-green-600"></div>
      </td>
    </tr>
  </tbody>
</table>
</div>

---

# Revocação
Observação

> Alta revocação é importante quando o custo de um Falso Negativo é alto. Por exemplo, diagnóstico de câncer. É melhor ter um falso alarme (FP) do que deixar um paciente doente sem diagnóstico (FN).

---
layout: quote
---

# F1-Score

> A média harmônica entre Precisão e Revocação. É uma ótima métrica para avaliar o equilíbrio entre as duas.

$$F1 = 2 \cdot \frac{Precisão \cdot \text{Revocação}}{Precisão + \text{Revocação}}$$

---
layout: quote
---

# F1-Score
Observação


> O F1-Score penaliza modelos que são bons em uma métrica mas ruins na outra. É uma das métricas mais usadas para problemas de classificação, especialmente com classes desbalanceadas.

---
layout: section
---

# 6\. Algoritmos de Classificação

---
layout: section
---

<img src="/c5_10.png" class="mx-auto" />

---
layout: quote
---

# Árvore de Decisão

> Um modelo não-paramétrico que aprende regras de decisão simples inferidas a partir das características dos dados. É um modelo intuitivo, semelhante a um fluxograma.

---

# Árvore de Decisão
Componentes

- **Nó Raiz (*Root Node*)**: O ponto de partida, representa todo o dataset.
- **Nó de Decisão (*Decision Node*)**: Um nó que se divide em outros nós.
- **Nó Folha (*Leaf Node*)**: Um nó terminal que representa uma classe final.
- **Divisão (*Splitting*)**: O processo de dividir um nó em sub-nós.

---
layout: quote
---

# Como a Árvore Aprende?

> O algoritmo busca, em cada nó, a melhor característica e o melhor limiar para dividir os dados de forma que os sub-grupos resultantes sejam os mais "puros" possíveis em termos de classe.

---
layout: fact
---

# Mas, o que é "pureza"?

---
layout: quote
---

> A pureza é determinada através de métricas como o **Índice de Gini** ou a **Entropia**. Essas métricas medem a desigualdade entre os dados.

---
layout: quote
---

# Índice de Gini

> Mede a probabilidade de um elemento, escolhido aleatoriamente, ser classificado incorretamente.

---
layout: quote
---

# Índice de Gini

$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$

- $C$: número de classes.
- $p_i$: proporção de amostras da classe $i$ no nó.

> **Gini = 0** Pureza máxima, todas as amostras no nó pertencem a uma única classe. Menor desigualdade

> **Gini = 1** Maior estradificação entre as classes. Maior desigualdade.

---
layout: quote
---

# Critérios de Divisão
Índice de Gini

> O algoritmo escolhe a divisão que resulta na maior **redução da impureza Gini**.

---
layout: quote
---

# Critérios de Divisão
Entropia

> Mede o grau de desordem ou incerteza em um nó.

$$Entropia (H) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$

  - **Entropia = 0**: Pureza máxima.
  - **Entropia = 1** (para 2 classes): Impureza máxima.

> O algoritmo usa a Entropia para calcular o **Ganho de Informação**, e escolhe a divisão que maximiza esse ganho.

---
layout: quote
---

# Árvores de Decisão
Vantagens

- **Interpretável**: Fácil de visualizar e entender as regras de decisão.
- **Pouca preparação de dados**: Não requer normalização de características.
- **Lida com dados numéricos e categóricos**.
- **Não-paramétrico**: Não faz suposições sobre a distribuição dos dados.

---
layout: quote
---

# Árvores de Decisão
Desvantagens

- **Propenso a *overfitting***: Pode criar árvores excessivamente complexas que decoram o ruído dos dados de treino. (Solução: Poda ou uso em Ensembles).
- **Instável**: Pequenas variações nos dados podem resultar em uma árvore completamente diferente.

---
layout: quote
---

# Árvore de Decisão com `scikit-learn`
Exemplo

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Criando o classificador com critério Gini e profundidade máxima para evitar overfitting
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
```

---
layout: section
---

# *Naive Bayes*

---
layout: quote
---

# Naive Bayes

> Uma família de classificadores probabilísticos baseados no **Teorema de Bayes**, com uma suposição "ingênua" (*naive*) de independência condicional entre as características.

> É particularmente eficaz para problemas de classificação de texto (ex: filtros de spam).

---
layout: quote
---

# Teorema de *Bayes*

> Descreve a probabilidade de um evento, com base no conhecimento prévio de condições que podem estar relacionadas ao evento.

$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$

- $P(A|B)$: **Probabilidade Posterior** (probabilidade da hipótese A, dado a evidência B).
- $P(B|A)$: **Verossimilhança (Likelihood)** (probabilidade da evidência B, dado que A é verdade).
- $P(A)$: **Probabilidade a Priori** (probabilidade da hipótese A antes de ver a evidência).
- $P(B)$: **Probabilidade Marginal da Evidência**.

---

# Teorema de *Bayes*
Exemplos

> Teste para uma doença com 99% de eficácia sabendo que a prevalência dessa doença na população é de 0.1%. O Teorema de Bayes ajuda a calcular a probabilidade real de que um resultado positivo signifique que você tem a doença.

> Um sistema de recomendação de uma plataforma de *streaming* pode aplicar o Teorema de Bayes num algoritmo para ajustar suas recomendações à medida que mais dados são coletados sobre as preferências.

---
layout: quote
---

# Aplicando Bayes à Classificação

> Para um dado vetor de características $X = (x_1, x_2, ..., x_n)$, queremos encontrar a classe $C_k$ que maximiza a probabilidade posterior $P(C_k|X)$.

$$P(C_k | X) = \frac{P(X | C_k) \cdot P(C_k)}{P(X)}$$

- $P(C_k)$: Probabilidade a Priori da classe (frequência da classe nos dados de treino).
- $P(X | C_k)$: Verossimilhança das características, dada a classe.

---
layout: quote
---

# A Suposição "Ingênua"

> Calcular $P(X | C_k) = P(x_1, x_2, ..., x_n | C_k)$ é computacionalmente complexo. O Naive Bayes assume que **todas as características são condicionalmente independentes entre si**, dada a classe.

$$P(X | C_k) = P(x_1|C_k) \cdot P(x_2|C_k) \cdot ... \cdot P(x_n|C_k) = \prod_{i=1}^{n} P(x_i|C_k)$$

> Esta suposição simplifica drasticamente o cálculo, tornando o algoritmo muito rápido.

---
layout: quote
---

# Naive Bayes
Vantagens

- **Extremamente rápido** para treinar e prever.
- **Requer poucos dados** de treinamento para uma performance razoável.
- Funciona bem com **alta dimensionalidade** (muitas características).
- Ótimo para classificação de texto.

---
layout: quote
---

# Naive Bayes
Desvantagens

- A suposição de independência das características é muitas vezes **irrealista**.
- Performance pode ser inferior a modelos mais complexos quando as características são correlacionadas.

---
layout: quote
---

# Naive Bayes com `scikit-learn`
Exemplo

```python
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# GaussianNB é usado para características contínuas que seguem uma distribuição normal
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
```

---
layout: section
---

# K-Nearest Neighbors (KNN)

<img src="/c5_11.png" class="w-2/3 mx-auto" />

---
layout: quote
---

# K-Nearest Neighbors (KNN)

> Um algoritmo de aprendizado **baseado em instância** (ou *lazy learning*). Ele não aprende um modelo explícito, mas sim armazena todo o conjunto de dados de treinamento. A classificação de uma nova amostra é feita por um "voto majoritário" de seus vizinhos mais próximos.

**Intuição**: "Diga-me quem são seus vizinhos, e eu te direi quem você é."

---
layout: quote
---

# Como o KNN Funciona?

1.  **Escolha de K**: Defina o número de vizinhos (K) a serem considerados.
2.  **Cálculo de Distância**: Para uma nova amostra, calcule a distância entre ela e **todas** as amostras no conjunto de treinamento.
3.  **Encontre os Vizinhos**: Identifique os K vizinhos mais próximos (com as menores distâncias).
4.  **Votação**: Atribua à nova amostra a classe que é mais frequente entre os K vizinhos.

---
layout: quote
---

# A Importância de K

- **K pequeno** Limite de decisão mais flexível e complexo. Alta variância, baixo viés. **Sensível a ruído e outliers**.
- **K grande** Limite de decisão mais suave e simples. Baixa variância, alto viés. **Mais robusto a ruído**.

> A escolha de K é um hiperparâmetro crucial e geralmente é feita por meio de validação cruzada.

---
layout: quote
---

# Métricas de Distância

> A definição de "proximidade" depende da métrica de distância usada.

---

# Distância Euclidiana 

> A distância em linha reta entre dois pontos (mais comum)

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

---
layout: quote
---

# Métricas de Distância

> **Distância de Manhattan** A soma das diferenças absolutas das coordenadas.

$$d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

---

# Visualização

```bash{*}{lines: false}
A ----3---- B
|           |
|           |
4           4  
|           |
|           |
C ----3---- D
```

Entre A e D
- Distância Euclidiana: linha reta diagonal = $\sqrt{3² + 4^2} = 5$
- Distância de Manhattan: $3 + 4 = 7$ (como caminhar pelas ruas)

---
layout: quote
---

# KNN
Observação

> Como o KNN é baseado em distância, é fundamental que as características estejam na mesma escala. **A normalização ou padronização dos dados é quase sempre necessária.**


---
layout: quote
---

# KNN
Vantagens

- Simples de entender e implementar.
- **Não faz suposições** sobre os dados (não-paramétrico).
- **Eficaz** quando as características são marcantes e os dados não forem muito ruidosos.

---
layout: quote
---

# KNN
Desvantagens

- **Custo computacional** A fase de previsão é lenta em *datasets* grandes, pois demanda cálculos de distância para todos os pontos de treino.
- **A Maldição da Dimensionalidade** A performance degrada rapidamente com o aumento do número de características
- Sensível a características **irrelevantes** e à **escala** dos dados

---
layout: quote
---

# KNN com `scikit-learn`
Exemplo

```python{*}{class:'!children:text-xs'}
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_scoreq

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 1. Escalonar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 2. Criar e treinar o modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
# 3. Prever e avaliar
y_pred = knn.predict(X_test_scaled)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
```

---
layout: section
---

# Support Vector Machine (SVM)

<img src="/c5_12.png" class="w-2/3 mx-auto" />

---
layout: quote
---

# SVM
*Support Vector Machine*

> Um classificador poderoso que busca encontrar o **hiperplano** que melhor separa as classes no espaço de características. O "melhor" hiperplano é aquele que tem a **maior margem** entre ele e os pontos de dados mais próximos de cada classe.

---
layout: quote
---

# Conceitos Fundamentais

- **Hiperplano**: O limite de decisão. Em 2D, é uma linha. Em 3D, é um plano. Em mais dimensões, é um hiperplano.
- **Margem**: A distância entre o hiperplano e os pontos de dados mais próximos (de qualquer classe). O SVM tenta maximizar essa margem.
- **Vetores de Suporte (Support Vectors)**: Os pontos de dados que estão na borda da margem. São os pontos mais difíceis de classificar e os únicos que "suportam" ou definem o hiperplano.

---
layout: quote
---

# O Truque do Kernel (*Kernel Trick*)

> E se os dados não forem linearmente separáveis? O SVM pode usar o **truque do kernel** para mapear os dados de entrada para um espaço de características de dimensão mais alta, onde eles se tornam linearmente separáveis.

- *Kernels* típicos
  - **Linear**: `kernel='linear'`
  - **Polinomial**: `kernel='poly'`
  - **Gaussiano (RBF - Radial Basis Function)**: `kernel='rbf'` (o mais popular e poderoso).

---
layout: two-cols-header
---

# SVM - RBF
Radius Base Function

:: left ::

- Fronteiras de decisão não lineares

:: right ::

<img src="/c5_13.png" class="w-1/2 mx-auto" />

---

# Visualizando o Truque do Kernel

> Mapeando para uma dimensão superior para encontrar um hiperplano separador.

<img src="/c5_14.png" class="w-3/4 mx-auto" />

---
layout: quote
---

# SVM
Vantagens

- Eficaz em espaços de **alta dimensão**
- **Versátil** graças aos diferentes *kernels*
- **Robusto a *overfitting*** em alta dimensão
- Funciona bem com **limites de decisão claros**, mesmo que não sejam lineares

---
layout: quote
---

# SVM
Desvantagens

- Ineficiente em *datasets*  **grandes** (complexidade computacional)
- Sensível à escolha do *kernel* e seus hiperparâmetros (como $C$ e $gamma$).
- **Menos interpretável** comparado às Árvores de Decisão.

---
layout: quote
---

# SVM com `scikit-learn`
Exemplo

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Usando o kernel RBF, que é poderoso para limites não-lineares
svc = SVC(kernel='rbf', C=1.0, gamma='auto')
svc.fit(X_train_scaled, y_train)

y_pred = svc.predict(X_test_scaled)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
```

---

# SVM com `scikit-learn`
Outros Exemplos

<br>

[SVM com Diferentes Kernels](https://colab.research.google.com/drive/1OaCyw6l-Ibdu5VqP48Gpek_aBw31DGEk?usp=sharing)

<br>

[SVM Comparativo de Classes e Features](https://colab.research.google.com/drive/1REp7EkAtbsiYJJNpYjNqNEX-OzUfClFz?usp=sharing)

---
layout: section
---

# 7\. Métodos Ensemble

---
layout: quote
---

# A Sabedoria da Multidão

> Métodos Ensemble combinam as previsões de vários modelos de aprendizado de máquina (chamados de "weak learners" ou "base models") para produzir uma previsão mais precisa, robusta e estável do que qualquer modelo individual.

> Dois tipos principais: **Bagging** e **Boosting**.

---
layout: quote
---

# Bagging (Bootstrap Aggregating)

> A ideia é reduzir a **variância** de modelos que tendem a superajustar (como Árvores de Decisão profundas).

1.  Cria múltiplos subconjuntos de dados do dataset de treino original por amostragem com reposição (**Bootstrap**).
2.  Treina um modelo independente em cada subconjunto.
3.  Agrega as previsões de todos os modelos (voto majoritário para classificação, média para regressão) - **Aggregating**.

---
layout: quote
---

# Random Forest

> É um método de **Bagging** onde os modelos base são **Árvores de Decisão**. Além do bootstrap, em cada divisão da árvore, ele considera apenas um subconjunto aleatório de características. Isso descorrelaciona as árvores e torna o ensemble ainda mais robusto

> É um dos algoritmos mais poderosos e amplamente utilizados em *Machine Learning*

---
layout: quote
---

# Boosting

> A ideia é reduzir o **viés**, transformando uma coleção de modelos fracos em um único modelo forte.

1.  Os modelos são treinados **sequencialmente**.
2.  Cada novo modelo foca em corrigir os erros do modelo anterior. As amostras que foram classificadas incorretamente recebem um peso maior no treinamento do próximo modelo.
3.  As previsões finais são uma soma ponderada de todos os modelos.

**Exemplos famosos**: AdaBoost, Gradient Boosting, XGBoost, LightGBM.

---
layout: quote
---

# Bagging vs. Boosting

Paralelo vs. Sequencial

<img src="/c5_15.png" class="w-3/4 mx-auto" />

---
layout: quote
---

# Exemplo: Random Forest com `scikit-learn`

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# n_estimators é o número de árvores na floresta
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")

# Random Forest também pode nos dar a importância de cada característica
importances = rf_clf.feature_importances_
```

---
layout: section
---

# 8\. Conclusão

---

# Fluxo de Trabalho de Aprendizado Supervisionado

1.  **Definição do Problema**: É regressão ou classificação?
2.  **Coleta e Preparação dos Dados**: Limpeza, tratamento de valores ausentes, engenharia de características, escalonamento.
3.  **Divisão dos Dados**: Treino, Validação e Teste.
4.  **Seleção e Treinamento do Modelo**: Escolha um ou mais algoritmos e treine-os com os dados de treino.
5.  **Avaliação do Modelo**: Use as métricas apropriadas no conjunto de teste.

---

6.  **Ajuste de Hiperparâmetros**: Otimize os parâmetros do modelo (ex: `K` no KNN, `C` no SVM).
7.  **Finalização e Implantação**: Treine o modelo final com os dados e use-o para previsões em dados novos.

---
layout: quote
---

# Resumo

- **Aprendizado Supervisionado**: Aprender de dados rotulados.
- **Regressão**: Prever números. Avaliado com MAE, RMSE, R².
- **Classificação**: Prever categorias. Avaliado com Acurácia, Precisão, Recall, F1-Score, Matriz de Confusão.

---
layout: quote
---

# Resumo

- **Algoritmos**: Cada um com sua própria intuição, matemática e *tradeoffs*.
    - **Árvore de Decisão**: Interpretável, mas propenso a overfitting.
    - **Naive Bayes**: Rápido e simples, com uma forte suposição de independência.
    - **KNN**: Baseado em distância, "preguiçoso", sensível à escala.
    - **SVM**: Encontra a margem máxima, poderoso com kernels.
- **Ensembles**: Combinar modelos para obter melhores resultados (Random Forest é um exemplo poderoso).

---
layout: fact
---

# Dúvidas

---

# Referências

- [KNN](https://www.ibm.com/think/topics/knn)
- [Naive Bayes](https://www.ibm.com/think/topics/naive-bayes)
- [Naive Bayes - Exemplo](https://codinginfinite.com/naive-bayes-classification-numerical-example)
- [Ensemble](https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms)
- [Documentação Scikit-Learn](https://scikit-learn.org/stable)

---

# Exercícios Resolvidos

- [Regressão Linear¹](https://colab.research.google.com/drive/1bLkkzVUge6CBRspE8h8TOAHls2mAjUJ7)
- [Regressão Linear²](https://colab.research.google.com/drive/1z7ZV2jPsNEe2p2Qxjwv3bTZkf4bnLA2_)
- [Árvores de Decisão](https://colab.research.google.com/drive/1XqvB7vUQmNqi00i-ildPyk4Z40aT9dXA)
- [Naive Bayes](https://colab.research.google.com/drive/1JHHy23Rvbjv8OAopjOVRoNKwnxgggUC5)
- [KNN](https://colab.research.google.com/drive/1A80fBO6oKGhteWchIDBxWY6w7_EKTEy-)
- [SVM](https://colab.research.google.com/drive/1ep5iXzRaZqFTvFzAzeDduC9900qWzbH0)
- [Ensemble](https://colab.research.google.com/drive/1G1uGwAkHXf2BhLc0pWB15cqBqeu1wArg)

---
src: /snippets/end.md
---