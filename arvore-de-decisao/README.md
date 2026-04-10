# Árvore de Decisão — Previsão de Churn em Telecomunicações

Implementação de árvores de decisão (ID3, C4.5 e CART) para classificação do status de clientes de uma operadora de telecomunicações. Todo o pipeline está em [`model.ipynb`](model.ipynb) e utiliza o dataset [`data/telecom_customer_churn.csv`](data/telecom_customer_churn.csv).

---

## 1. O problema

Empresas de telecomunicações convivem com um problema caro: a **rotatividade de clientes (churn)**. Identificar com antecedência quais clientes tendem a cancelar o serviço permite ações direcionadas de retenção (ofertas, contato ativo, ajuste de plano), que têm custo muito menor do que adquirir um novo cliente.

Neste projeto, o objetivo é **prever o status do cliente** a partir de seus dados cadastrais, de uso do serviço e de faturamento. A variável-alvo é `Customer Status`, uma classificação multi-classe com três categorias:

- **Stayed** — cliente ativo que permaneceu.
- **Churned** — cliente que cancelou o serviço.
- **Joined** — cliente recém-adquirido.

A escolha pela **árvore de decisão** como modelo não é arbitrária:

- **Interpretabilidade**: uma árvore gera regras explícitas do tipo *"se `Contract = Month-to-Month` e `Tenure in Months <= 6` então provavelmente Churned"*. Isso é fundamental num contexto de negócio em que o time de retenção precisa justificar e agir sobre a decisão.
- **Heterogeneidade do dataset**: o conjunto mistura variáveis numéricas (idade, receita, meses de contrato) e categóricas (tipo de contrato, forma de pagamento, serviços contratados). Árvores lidam naturalmente com ambos os tipos.
- **Pouca necessidade de normalização**: por particionarem o espaço com limiares, árvores não exigem padronização de escala.

---

## 2. O dataset

| Característica | Valor |
|---|---|
| Fonte | `data/telecom_customer_churn.csv` |
| Registros | 7.043 clientes |
| Atributos originais | 38 colunas |
| Variável-alvo | `Customer Status` (3 classes) |

O dataset traz informações demográficas (`Age`, `Gender`, `Married`, `Number of Dependents`), geográficas (`City`, `Zip Code`, `Latitude`, `Longitude`), comerciais (`Offer`, `Contract`, `Payment Method`), de uso dos serviços (`Phone Service`, `Internet Type`, `Streaming TV`, etc.) e de faturamento (`Monthly Charge`, `Total Charges`, `Total Revenue`).

---

## 3. Pipeline da implementação

O notebook está organizado em 10 seções, seguindo um fluxo completo de modelagem supervisionada:

```
1. Importação de bibliotecas
2. Análise exploratória inicial do dataset
3. Tratamento de dados (nulos, duplicados)
4. Análise Exploratória de Dados (EDA)
5. Feature Engineering (remoção, encoding, seleção, split)
6. Árvore ID3 (entropia, sem poda)
7. Árvore C4.5 (entropia + Cost-Complexity Pruning)
8. Comparação entre 4 configurações (ID3, C4.5, CART, CART podado)
9. Discussão sobre overfitting, poda e validação
10. Interpretação do modelo (importâncias e regras)
```

A seguir, detalhamos cada etapa relevante.

---

## 4. Tratamento de dados

### 4.1 Análise dos valores nulos

A primeira inspeção mostra blocos coerentes de nulos:

- As colunas de serviços de internet (`Internet Type`, `Online Security`, `Online Backup`, `Device Protection Plan`, `Premium Tech Support`, `Streaming TV/Movies/Music`, `Unlimited Data`, `Avg Monthly GB Download`) estão nulas **exatamente** para os clientes cujo `Internet Service = No`. Ou seja, **não são dados faltantes reais — representam ausência do serviço**.
- O mesmo acontece com `Multiple Lines` e `Avg Monthly Long Distance Charges`, nulos quando `Phone Service = No`.
- `Churn Category` e `Churn Reason` só estão preenchidas para clientes que já churnaram — essas colunas **explicam** a variável alvo e, se mantidas, causariam **data leakage** (o modelo acertaria 100% trivialmente).

### 4.2 Decisões de limpeza

```python
# Remoção de colunas que causariam leakage ou são identificadores
df.drop(columns=['Customer ID', 'Churn Category', 'Churn Reason'], inplace=True)

# Nulos de serviço de internet → categoria explícita "No Internet Service"
df[<colunas_internet>] = df[<colunas_internet>].fillna('No Internet Service')

# Nulos de telefone → categoria explícita "No Phone Service"
df[['Avg Monthly Long Distance Charges', 'Multiple Lines']] = \
    df[['Avg Monthly Long Distance Charges', 'Multiple Lines']].fillna('No Phone Service')

# Oferta ausente → categoria "No Offer"
df['Offer'] = df['Offer'].fillna('No Offer')
```

A estratégia **não** é preencher com média/mediana — como os nulos têm significado semântico, criar uma categoria própria ("No Internet Service") preserva a informação e permite que a árvore aprenda sobre ela.

---

## 5. Análise Exploratória (EDA)

A EDA é feita em 4 frentes:

1. **Distribuição da variável-alvo** — revela o desbalanceamento entre Stayed, Churned e Joined, calculando a taxa de churn.
2. **Histogramas das variáveis numéricas por status** — destaca que `Tenure in Months`, `Monthly Charge` e `Total Charges` têm distribuições visivelmente diferentes entre clientes que permaneceram e que churnaram.
3. **Matriz de correlação** — identifica pares altamente correlacionados (`Total Charges` × `Total Revenue` ≈ 0.99), apontando redundâncias.
4. **Proporção de cada variável categórica por status** — mostra que `Contract` (mensal vs. anual) tem forte poder discriminativo.

### Principais insights

- **Tenure in Months** é um dos mais fortes indicadores: churn se concentra em contratos recentes.
- **Contract**: clientes `Month-to-Month` churnam muito mais do que os de contrato anual.
- **Total Charges ≈ Total Revenue** → manter apenas uma evita multicolinearidade.
- **Latitude, Longitude, Zip Code, City**: alta cardinalidade e baixo poder preditivo para árvores (gerariam splits ruidosos) → **removidas**.
- **Phone Service** é redundante com `Multiple Lines` (onde "No Phone Service" já é uma categoria).
- **Internet Service** é redundante com `Internet Type` pelo mesmo motivo.

---

## 6. Feature Engineering

### 6.1 Remoção de colunas redundantes/irrelevantes

```python
cols_to_drop = [
    'City', 'Zip Code', 'Latitude', 'Longitude',  # geolocalização com alta cardinalidade
    'Total Revenue',                              # correlação ~0.99 com Total Charges
    'Phone Service',                              # redundante com Multiple Lines
    'Internet Service',                           # redundante com Internet Type
]
```

### 6.2 Correção de tipos mistos

`Avg Monthly Long Distance Charges` e `Avg Monthly GB Download` passaram a conter strings ("No Phone Service", "No Internet Service") após o `fillna`. São convertidas de volta para numérico com `pd.to_numeric(..., errors='coerce').fillna(0.0)` — ou seja, clientes sem o serviço recebem **0.0**, preservando a semântica.

### 6.3 Encoding das variáveis categóricas

O `DecisionTreeClassifier` do scikit-learn só aceita entrada numérica. Aplica-se `LabelEncoder` para cada coluna categórica, guardando os mapeamentos em `label_encoders` para poder interpretar a árvore depois. O target também é encodado.

> **Nota importante**: `LabelEncoder` impõe uma ordem arbitrária nos valores categóricos. Para árvores isso não é ideal (seria melhor usar one-hot para variáveis nominais), mas funciona bem aqui porque a árvore consegue criar splits do tipo `x <= k.5` que isolam valores específicos, e o foco do trabalho é comparar critérios de divisão e estratégias de poda, não otimizar encoding.

### 6.4 Seleção de features por correlação com o target

```python
corr_with_target = features_df.corr()['target'].abs().sort_values(ascending=False)
selected_features = corr_with_target[corr_with_target > 0.05].index.tolist()
```

Features com correlação absoluta menor que **0.05** com o target são descartadas. Isso reduz o espaço de busca da árvore, diminui overfitting e acelera o treino sem perder poder preditivo relevante.

### 6.5 Divisão treino/teste

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **20% para teste**, 80% para treino.
- **`stratify=y`**: garante que a proporção de cada classe seja preservada nos dois conjuntos — crucial num dataset desbalanceado.
- **`random_state=42`**: reprodutibilidade.

---

## 7. Os três algoritmos implementados

O trabalho compara os três algoritmos clássicos de árvores de decisão. No scikit-learn não existem implementações nativas do ID3 e do C4.5 — eles são **simulados** via `DecisionTreeClassifier` com configurações específicas.

### 7.1 ID3 (Iterative Dichotomiser 3)

**Teoria**: ID3 usa **entropia** como medida de impureza e **ganho de informação** como critério de escolha do atributo em cada nó.

- **Entropia**: `H(S) = -Σ pᵢ · log₂(pᵢ)`, onde `pᵢ` é a proporção da classe `i` em `S`. Mede desordem: 0 = conjunto puro, máximo = classes perfeitamente distribuídas.
- **Ganho de Informação**: `IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) · H(Sᵥ)`. Quanto maior, melhor o atributo `A` separa as classes.
- **Critério de parada (puro ID3)**: a árvore cresce até que todas as folhas sejam puras ou os atributos se esgotem. **Sem poda**.

**Simulação no scikit-learn**:

```python
tree_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
tree_id3.fit(X_train, y_train)
```

Usamos `criterion='entropy'` (que implementa ganho de informação) e **nenhuma restrição** (`max_depth`, `min_samples_leaf`, `ccp_alpha` todos no default permissivo). O resultado é uma árvore extremamente profunda com ~100% de acurácia no treino e queda significativa no teste — o clássico **overfitting**.

### 7.2 C4.5

**Teoria**: Evolução do ID3 com melhorias importantes:

- **Tratamento de atributos contínuos**: avalia todos os limiares possíveis (pontos médios entre valores ordenados) e escolhe o que maximiza o ganho. O CART do scikit-learn já faz isso nativamente.
- **Poda pós-construção**: constrói a árvore completa e depois remove ramos que não contribuem significativamente para a acurácia, reduzindo overfitting.
- **Gain Ratio** (opcional): normaliza o ganho de informação pela entropia intrínseca do atributo, penalizando atributos com muitos valores — não implementado aqui pois o scikit-learn não oferece Gain Ratio.

**Simulação no scikit-learn**: entropia + **Cost-Complexity Pruning (CCP)** via parâmetro `ccp_alpha`.

```python
tree_c45 = DecisionTreeClassifier(
    criterion='entropy',
    ccp_alpha=best_alpha,     # obtido por validação cruzada (ver 7.4)
    random_state=42
)
```

### 7.3 CART (Classification and Regression Trees)

**Teoria**: Usa o **Índice de Gini** como critério de impureza.

- **Gini**: `G(S) = 1 - Σ pᵢ²`. Mede a probabilidade de classificar incorretamente um elemento escolhido aleatoriamente. Também varia de 0 (puro) a um máximo dependente do número de classes.
- Computacionalmente mais barato que entropia (evita `log₂`), com resultados tipicamente similares.
- Sempre gera árvores binárias.

**Implementação**: `criterion='gini'`, com duas variantes — sem poda e com `ccp_alpha` ótimo.

### 7.4 Escolha do `ccp_alpha` ótimo (Cost-Complexity Pruning Path)

A parte mais cuidadosa da implementação está em como escolher **quanto** podar. O processo tem 3 passos:

**1. Obter o caminho de poda completo**

```python
path = tree_id3.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
```

Isso retorna a sequência de valores de alpha que geram subárvores estruturalmente distintas — cada alpha corresponde a um nível de poda específico.

**2. Traçar a curva acurácia × alpha**

Para cada alpha candidato, treina-se uma árvore e mede-se acurácia de treino e teste. A curva resultante evidencia o tradeoff viés-variância: alphas baixos geram árvores complexas (overfitting), alphas altos geram árvores simples demais (underfitting).

**3. Selecionar via validação cruzada 5-fold**

Escolher o melhor alpha olhando diretamente o conjunto de teste seria **data leakage** — o teste deixaria de ser imparcial. A solução correta é usar **cross-validation só no treino**:

```python
for alpha in alpha_candidates:
    clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    cv_means.append(scores.mean())

best_alpha = alpha_candidates[np.argmax(cv_means)]
```

O alpha que maximiza a acurácia média do CV é usado para treinar a árvore final, e **só então** o teste é usado para medir o desempenho real de generalização.

O mesmo procedimento é repetido para o critério Gini, gerando a variante "CART podado".

---

## 8. Comparação entre configurações

O trabalho avalia quatro configurações lado a lado:

| Config | Critério | Poda                   | Simula        |
| ------ | -------- | ---------------------- | ------------- |
| A      | Entropy  | Sem                    | ID3           |
| B      | Entropy  | Com (`ccp_alpha` ótimo) | C4.5          |
| C      | Gini     | Sem                    | CART          |
| D      | Gini     | Com (`ccp_alpha` ótimo) | CART podado   |

Para cada modelo são computadas:

- **Acurácia no treino** — detecta overfitting quando muito alta.
- **Acurácia no teste** — mede generalização.
- **F1 Score Macro** — métrica balanceada entre classes (importante no desbalanceamento).
- **Profundidade** e **número de folhas** — medem complexidade estrutural.

### 8.1 Curvas de validação por profundidade

Independente do CCP, também se varia `max_depth` de 1 a 30 para entropy e gini, plotando a curva treino-vs-teste. A curva mostra visualmente:

- **Profundidades baixas** → underfitting (ambas as acurácias baixas).
- **Profundidades altas** → treino sobe para 100%, teste estagna ou cai (overfitting).
- **Ponto ótimo** → máximo da curva de teste.

Esse é o retrato canônico do **tradeoff viés-variância**.

---

## 9. Discussão: Overfitting, Poda e Validação

**Por que overfitting acontece em árvores?** Sem restrições, a árvore cresce até que cada folha seja pura. Isso significa decorar os dados de treino, inclusive ruído. O resultado é regras muito específicas que não generalizam.

**Como a poda resolve?** Existem duas estratégias:

- **Pré-poda**: limitar o crescimento durante a construção via `max_depth`, `min_samples_split`, `min_samples_leaf`. Simples, mas exige testar valores no escuro.
- **Pós-poda (Cost-Complexity)**: construir a árvore inteira e depois remover ramos que não compensam sua complexidade. `ccp_alpha` controla esse tradeoff — alpha maior significa penalização maior e árvore menor. É a abordagem escolhida neste trabalho, por ser a mais alinhada com o C4.5 e com fundamentação teórica (cost-complexity pruning de Breiman).

**Por que validação cruzada?** Escolher hiperparâmetros (`ccp_alpha`, `max_depth`) olhando o conjunto de teste enviesa a avaliação final. A **5-fold cross-validation no treino** permite estimar o desempenho de cada configuração sem tocar no teste, garantindo que o teste continue sendo uma estimativa imparcial da generalização.

---

## 10. Interpretação do modelo

### 10.1 Feature Importances

```python
importances = tree_c45.feature_importances_
```

O scikit-learn calcula a importância de cada feature como a **redução total (ponderada pelo tamanho do nó) da impureza** que ela proporciona somada em toda a árvore. É uma medida direta de quanto cada atributo contribui para a capacidade discriminativa do modelo.

### 10.2 Regras extraídas

```python
rules = export_text(tree_c45, feature_names=selected_features, class_names=le_target.classes_, max_depth=5)
```

`export_text` imprime a árvore como regras if-then legíveis:

```
|--- Contract <= 0.5
|   |--- Tenure in Months <= 6.5
|   |   |--- class: Churned
|   |--- Tenure in Months > 6.5
|   |   |--- ...
```

Essas regras podem ser diretamente comunicadas ao time de negócio.

### 10.3 Limiares para atributos contínuos

Para cada atributo numérico, o scikit-learn testa todos os pontos médios entre valores consecutivos ordenados e escolhe o limiar que maximiza o critério (entropia ou Gini) — comportamento equivalente ao C4.5. Por isso os nós mostram condições como `Tenure in Months <= 6.5`: o algoritmo descobriu que **6 meses é o ponto de corte ótimo** para separar os grupos naquele ramo.

---

## 11. Como executar

**Requisitos**:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Rodar o notebook**:

```bash
cd arvore-de-decisao
jupyter notebook model.ipynb
```

Ou abrir diretamente no VSCode com a extensão de Jupyter. Basta executar as células na ordem — o notebook é totalmente reprodutível (`random_state=42` em todos os pontos críticos).

---

## 12. Estrutura de arquivos

```
arvore-de-decisao/
├── README.md                       # este arquivo
├── model.ipynb                     # notebook com todo o pipeline
└── data/
    └── telecom_customer_churn.csv  # dataset (7043 clientes × 38 colunas)
```
