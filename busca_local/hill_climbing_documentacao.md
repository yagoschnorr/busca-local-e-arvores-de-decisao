# Torre de Transmissão — Hill Climbing

## O Problema

Queremos posicionar uma **torre de transmissão** (ou centro de distribuição) de forma a **minimizar a soma das distâncias euclidianas** até três cidades fixas.

$$f(x, y) = \sum_{i=1}^{3} \sqrt{(x - x_i)^2 + (y - y_i)^2}$$

Isso é o **Problema de Weber** (também chamado de Problema de Fermat-Torricelli generalizado): encontrar o ponto que minimiza a soma das distâncias a um conjunto de pontos fixos.

---

## O que é Hill Climbing?

**Hill Climbing** (ou *descida de encosta*, em minimização) é uma **metaheurística de busca local** simples e intuitiva:

```
1. Começa em uma solução s₀
2. Repete até convergir:
   a. Gera vizinhos de s atual
   b. Avalia a função objetivo em cada vizinho
   c. Move para um vizinho melhor (se existir)
   d. Para se nenhum vizinho melhora
```

### Analogia

Imagine que você está numa paisagem montanhosa com os olhos vendados e quer chegar ao **ponto mais baixo** (mínimo). Hill Climbing faz você sentir o terreno ao redor e dar um passo sempre "para baixo". Se em todas as direções o terreno sobe, você para — mas pode não estar no vale mais fundo!

---

## Variantes Implementadas

### 1. Simple Hill Climbing

Aceita o **primeiro vizinho melhor** encontrado, sem avaliar os demais.

```python
for angulo in angulos_aleatorios:
    vizinho = pos_atual + passo * [cos(angulo), sin(angulo)]
    if f(vizinho) < f(pos_atual):
        pos_atual = vizinho
        break  # ← para aqui, sem avaliar os outros vizinhos
```

**Vantagem:** Rápido — não precisa avaliar todos os vizinhos  
**Desvantagem:** Pode fazer movimentos subótimos

---

### 2. Steepest Ascent Hill Climbing (Descida mais íngreme)

Avalia **todos os vizinhos** e aceita o **melhor** entre eles.

```python
melhor = None
for angulo in angulos_fixos:   # direções igualmente espaçadas
    vizinho = pos_atual + passo * [cos(angulo), sin(angulo)]
    if f(vizinho) < f(melhor):
        melhor = vizinho

if melhor não é None:
    pos_atual = melhor         # aceita o MELHOR vizinho
else:
    para                       # ótimo local
```

**Vantagem:** Cada passo é o melhor possível dentro da vizinhança  
**Desvantagem:** Mais chamadas à função objetivo por iteração

---

### 3. Stochastic Hill Climbing

Gera **um vizinho aleatório** por iteração e aceita apenas se for melhor.

```python
perturbação = uniforme(-passo, +passo)  # ← aleatoriedade contínua
vizinho = pos_atual + perturbação
if f(vizinho) < f(pos_atual):
    pos_atual = vizinho
```

**Vantagem:** Exploração mais variada do espaço de busca  
**Desvantagem:** Convergência mais lenta; pode parar prematuramente por critério de estagnação

---

## Comparação com Simulated Annealing

| Característica | Hill Climbing | Simulated Annealing |
|---|---|---|
| Aceita pioras? | ❌ Nunca | ✅ Com probabilidade $e^{-\Delta f / T}$ |
| Risco de ótimo local | ⚠️ Alto | 🔵 Baixo (com T alto) |
| Velocidade de convergência | ⚡ Rápida | 🐢 Mais lenta |
| Qualidade da solução | Depende da inicialização | Geralmente melhor |
| Determinismo | Depende da variante | Estocástico |

### Quando usar Hill Climbing?

- A função é **convexa** (um único mínimo global)
- Você precisa de uma solução **rápida**, mesmo que não ótima
- Pode executar **múltiplas inicializações** para mitigar ótimos locais

### Quando preferir Simulated Annealing?

- A função tem **múltiplos ótimos locais**
- Você precisa de **maior qualidade** de solução
- O custo computacional por iteração é aceitável

---

## Problema dos Ótimos Locais

A função objetivo deste problema é **convexa** (soma de normas euclidianas), então Hill Climbing encontra o ótimo global a partir de **qualquer ponto inicial**.

Para ilustrar o problema dos ótimos locais em geral, considere uma função multimodal como:

$$g(x) = \sin(x) + 0.1x^2$$

Nesse caso, Hill Climbing pode parar em qualquer mínimo local dependendo do ponto inicial.

**Estratégias para lidar com ótimos locais:**
1. **Múltiplas reinicializações aleatórias** (Random Restarts Hill Climbing)
2. **Simulated Annealing** — aceita pioras com probabilidade decrescente
3. **Algoritmos Genéticos** — mantém uma população de soluções
4. **Busca Tabu** — proíbe revisitar soluções recentes

---

## Parâmetros Importantes

### Tamanho do Passo (`passo`)

- **Passo grande:** Exploração ampla, mas pode "saltar" sobre o ótimo
- **Passo pequeno:** Convergência fina, mas lenta e propensa a ótimos locais

**Estratégia adaptativa:** começar com passo grande e reduzir progressivamente (similar ao resfriamento do SA).

### Número de Vizinhos (`n_vizinhos`)

No **Simple** e **Steepest HC**, define quantas direções são avaliadas:

- Poucas direções: mais rápido, mas pode perder a direção ótima
- Muitas direções: mais preciso, mas mais lento

### Critério de Parada

- **Sem melhora:** Para quando nenhum vizinho é melhor (simples, mas pode parar cedo)
- **Máximo de iterações:** Garante término, mas pode parar antes de convergir
- **Estagnação:** Para após N iterações sem melhora (Stochastic HC)

---

## Resultados Esperados

Para o problema das três cidades nas posições `(1,2)`, `(5,8)` e `(9,3)`:

| Método | x ótimo | y ótimo | f mínimo |
|---|---|---|---|
| Referência (L-BFGS-B) | ≈ 5.22 | ≈ 4.84 | ≈ 12.4583 |
| Simple HC | ≈ 5.22 | ≈ 4.84 | ≈ 12.46 |
| Steepest HC | ≈ 5.22 | ≈ 4.84 | ≈ 12.46 |
| Stochastic HC | ≈ 5.22 | ≈ 4.84 | ≈ 12.46 |

> Como a função é convexa, todos os métodos convergem ao ótimo global.  
> Para funções multimodais, os resultados seriam mais variados.

---

## Estrutura dos Notebooks

```
├── simulated_annealing_torre.ipynb   ← SA original (problema base)
└── hill_climbing_torre.ipynb         ← HC com 3 variantes (este notebook)
```

### Como executar

```bash
# Instalar dependências
pip install numpy matplotlib scipy

# Abrir Jupyter
jupyter notebook hill_climbing_torre.ipynb
```

---

## Referências

- **Russell & Norvig** — *Artificial Intelligence: A Modern Approach*, Cap. 4 (Local Search)
- **Mitchell** — *An Introduction to Genetic Algorithms*
- **Kirkpatrick et al. (1983)** — *Optimization by Simulated Annealing*, Science
- **Weber, A. (1909)** — *Über den Standort der Industrien* (Problema de Weber original)
