# Guia de Geração de Contrafactuais para DCOPF

Este guia explica como usar a geração de contrafactuais com a nova arquitetura ICNN simplificada para problemas DC-OPF.

---

## O que são Contrafactuais?

**Contrafactuais** respondem à pergunta: *"O que precisa mudar na entrada para obter uma saída desejada?"*

### Exemplo em DC-OPF

**Situação atual:**
- Demanda: `x = [P1, P2, ..., Pn, Q1, Q2, ..., Qn]`
- Custo previsto: `y = NN(x) = $15,000`

**Pergunta contrafactual:**
- *"Qual demanda `x'` próxima de `x` resultaria em custo de $12,000?"*

**Resposta:**
- Demanda modificada: `x' = [P1', P2', ..., Pn', Q1', Q2', ..., Qn']`
- Custo previsto: `NN(x') ≈ $12,000`
- Com **mínimas mudanças** em relação a `x`

---

## Formulação MIP

O problema de contrafactual é formulado como uma otimização Mixed-Integer Programming (MIP):

```
min  ||x - x_factual||₁ + λ * num_changed

s.t. |NN(x) - y_target| ≤ ε
     x ∈ [x_min, x_max]
     x_i = x_factual_i  para i ∈ imutáveis
```

Onde:
- `x`: Demanda contrafactual (variável de decisão)
- `x_factual`: Demanda atual (factual)
- `y_target`: Custo alvo desejado
- `ε`: Tolerância (ex: 0.01 em escala normalizada)
- `λ`: Peso de esparsidade (penaliza número de features mudadas)
- `num_changed`: Número de features que mudaram

### Restrições da Rede Neural

A ICNN é incorporada no MIP através de restrições lineares e ReLU:

```julia
# Primeira camada: z_0 = ReLU(W_0 * x + b_0)
z_0[j] >= W_0[j,:] * x + b_0[j]  # ReLU parte 1
z_0[j] >= 0                       # ReLU parte 2

# Camadas hidden: z_i = ReLU(W_i * z_{i-1})
z_i[j] >= W_i[j,:] * z_{i-1}     # ReLU
z_i[j] >= 0

# Última camada: y = W_n * z_{n-1} (linear)
y == W_n * z_{n-1}
```

**Convexidade garante:** Como W_i ≥ 0, a rede é convexa, facilitando a otimização MIP.

---

## Como Usar

### Passo 1: Treinar o Modelo ICNN

Primeiro, você precisa treinar um modelo ICNN nos dados DCOPF:

```bash
# Gerar dados
julia src/data/Generate_DCOPF.jl

# Treinar modelo
julia examples/train_dcopf.jl
```

Isso cria:
- `tmp/dcopf_experiment/best_model.bson` - Modelo treinado
- `test_systems/data_pglib_opf_case118_ieee.bson` - Dados

### Passo 2: Gerar Contrafactuais

Execute o script de exemplo:

```bash
julia examples/generate_counterfactual.jl
```

Ou use programaticamente:

```julia
using Pkg
Pkg.activate(".")

include("src/ICNN.jl")
using .ICNN

include("counterfactuals/model_loader.jl")
include("counterfactuals/algorithms/mip_counterfactual.jl")

# 1. Carregar modelo e dados
model = load_icnn_model("tmp/dcopf_experiment/best_model.bson")
dataset = prepare_dcopf_dataset("test_systems/data_case118.bson")

# 2. Selecionar amostra factual
x_factual = dataset.X_test[1, :]  # Primeira amostra de teste
y_current = predict(model, reshape(x_factual, 1, :))[1, 1]

# 3. Definir alvo (ex: reduzir custo em 5%)
y_target = y_current - 0.05

# 4. Gerar contrafactual
result = generate_counterfactual(
    model,
    x_factual,
    Float64(y_target);
    epsilon=0.01,           # Tolerância
    sparsity_weight=0.1,    # Peso de esparsidade
    time_limit=120.0        # Tempo limite (s)
)

# 5. Extrair resultado
if result[:status] in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
    x_counterfactual = result[:counterfactual]
    println("Distância: $(result[:distance])")
    println("Features mudadas: $(result[:num_changed])")
    println("Custo previsto: $(result[:prediction])")
end
```

---

## Parâmetros Importantes

### `epsilon` - Tolerância do Alvo

Controla quão próximo o resultado deve estar do alvo:

```julia
# Tolerância pequena (mais preciso, pode ser mais difícil)
epsilon = 0.001  # Diferença máxima: 0.1% em escala normalizada

# Tolerância média (balanceado)
epsilon = 0.01   # Diferença máxima: 1%

# Tolerância grande (mais fácil de satisfazer)
epsilon = 0.05   # Diferença máxima: 5%
```

**Recomendação:** Comece com `epsilon=0.01` e ajuste se necessário.

### `sparsity_weight` - Peso de Esparsidade

Controla quantas features devem mudar:

```julia
# Esparsidade baixa (mais features mudam, menor distância)
sparsity_weight = 0.01

# Esparsidade média (balanceado)
sparsity_weight = 0.1

# Esparsidade alta (menos features mudam, maior distância)
sparsity_weight = 1.0
```

**Trade-off:**
- ↓ `sparsity_weight` → ↓ distância, ↑ features mudadas
- ↑ `sparsity_weight` → ↑ distância, ↓ features mudadas

**Recomendação:** Comece com `0.1` e compare diferentes valores.

### `x_bounds` - Limites das Features

Define os limites para as features contrafactuais:

```julia
# Dados normalizados (padrão)
x_bounds = (0.0, 1.0)

# Dados com normalização diferente
x_bounds = (-3.0, 3.0)  # Ex: standardization

# Limites personalizados
x_bounds = (minimum(X_test), maximum(X_test))
```

**Importante:** Use os mesmos limites da normalização dos dados de treino!

### `time_limit` - Tempo Limite

```julia
time_limit = 60.0    # 1 minuto (rápido, pode não achar ótimo)
time_limit = 120.0   # 2 minutos (balanceado)
time_limit = 300.0   # 5 minutos (mais garantia de solução)
```

### `immutable_indices` - Features Imutáveis

Especifica features que **não podem** ser mudadas:

```julia
# Exemplo: features 1, 2, 3 são fixas (ex: localização de barras)
immutable_indices = [1, 2, 3]

# Nenhuma feature fixa
immutable_indices = Int[]
```

**Uso em DCOPF:**
- Barras de geração fixas
- Topology da rede
- Características físicas imutáveis

---

## Interpretando Resultados

### Status da Otimização

```julia
status = result[:status]

# Status possíveis:
MOI.OPTIMAL          # ✓ Solução ótima encontrada
MOI.FEASIBLE_POINT   # ✓ Solução viável encontrada (pode não ser ótima)
MOI.INFEASIBLE       # ✗ Problema infeasível (sem solução possível)
MOI.TIME_LIMIT       # ✗ Tempo limite atingido
:already_at_target   # ✓ Já está no alvo (distância = 0)
```

### Métricas

```julia
result[:distance]         # Distância L1: ||x' - x||₁
result[:num_changed]      # Número de features que mudaram
result[:changed_indices]  # Índices das features mudadas
result[:prediction]       # Valor previsto NN(x')
result[:solve_time]       # Tempo de otimização (segundos)
```

### Validação

Sempre valide o resultado com um forward pass direto:

```julia
x_cf = result[:counterfactual]
y_mip = result[:prediction]
y_direct = predict(model, reshape(x_cf, 1, :))[1, 1]

println("MIP prediction: $y_mip")
println("Direct forward: $y_direct")
println("Difference: $(abs(y_mip - y_direct))")

# Deve ser < 1e-4 (devido a precisão numérica)
```

---

## Casos de Uso

### 1. Redução de Custo

**Objetivo:** Encontrar demanda que reduz custo operacional

```julia
# Atual: $15,000
y_current = 15000.0  # (em escala denormalizada)

# Alvo: reduzir 10%
y_target = y_current * 0.9  # $13,500

# Normalizar antes de usar
y_target_norm = (y_target - scaler_Y[:mean]) / scaler_Y[:std]

result = generate_counterfactual_regression(
    model, x_factual, Float64(y_target_norm);
    epsilon=0.01, sparsity_weight=0.1
)
```

### 2. Análise de Sensibilidade

**Objetivo:** Entender quais demandas mais afetam o custo

```julia
# Testar diferentes reduções de custo
reductions = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%

results = []
for reduction in reductions
    y_target = y_current * (1 - reduction)
    result = generate_counterfactual(model, x_factual, y_target)
    push!(results, (reduction, result))
end

# Analisar: quais features mudam mais para cada cenário?
```

### 3. Planejamento de Demanda

**Objetivo:** Encontrar padrões de demanda viáveis para custo objetivo

```julia
# Custo máximo aceitável
max_cost = 12000.0

# Gerar múltiplos contrafactuais com diferentes esparsidades
for λ in [0.01, 0.1, 1.0]
    result = generate_counterfactual(
        model, x_factual, Float64(max_cost);
        sparsity_weight=λ
    )

    # Comparar: qual é mais realista/implementável?
end
```

### 4. Detecção de Padrões

**Objetivo:** Identificar quais barras/zonas são mais importantes

```julia
result = generate_counterfactual(...)

# Features que mais mudaram
changed = result[:changed_indices]
changes = abs.(result[:counterfactual][changed] .- x_factual[changed])

# Top 10 features
top_10 = sortperm(changes, rev=true)[1:10]
println("Barras mais sensíveis:")
for idx in top_10
    println("  Barra $idx: Δ = $(changes[idx])")
end
```

---

## Troubleshooting

### Problema: Nenhuma solução encontrada (INFEASIBLE)

**Possíveis causas:**
1. `y_target` muito distante de `y_current`
2. `epsilon` muito pequeno
3. Muitas features imutáveis
4. Limites `x_bounds` muito restritivos

**Soluções:**
```julia
# Aumentar tolerância
epsilon = 0.05  # em vez de 0.01

# Reduzir esparsidade (permitir mais mudanças)
sparsity_weight = 0.01  # em vez de 0.1

# Aumentar tempo limite
time_limit = 600.0  # 10 minutos

# Relaxar limites
x_bounds = (-5.0, 5.0)  # mais amplo
```

### Problema: Tempo limite atingido (TIME_LIMIT)

**Soluções:**
```julia
# Aumentar tempo
time_limit = 600.0

# Simplificar modelo (usar rede menor para debug)
model = FICNN(n_features, 1; hidden_sizes=[50, 50])

# Reduzir precisão
epsilon = 0.05  # mais tolerante
```

### Problema: Muitas features mudam

**Soluções:**
```julia
# Aumentar peso de esparsidade
sparsity_weight = 1.0  # ou 10.0

# Fixar features menos importantes
# (requer análise de importância prévia)
immutable_indices = [1, 2, 3, ..., 50]
```

### Problema: Distância muito grande

**Soluções:**
```julia
# Relaxar alvo
y_target = y_current - 0.03  # menor mudança

# Reduzir esparsidade
sparsity_weight = 0.01

# Aumentar tolerância
epsilon = 0.05
```

---

## Análise Avançada

### Comparação de Múltiplos Contrafactuais

```julia
# Gerar contrafactuais com diferentes configurações
configs = [
    ("Low sparsity", 0.01),
    ("Med sparsity", 0.1),
    ("High sparsity", 1.0)
]

results = Dict()
for (name, λ) in configs
    result = generate_counterfactual(
        model, x_factual, y_target;
        sparsity_weight=λ
    )
    results[name] = result
end

# Comparar
println("Config          | Distance | Changed | Cost")
println("-" * "="^50)
for (name, result) in results
    if result[:status] in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        @printf("%-15s | %8.3f | %7d | %6.2f\n",
               name, result[:distance], result[:num_changed],
               result[:prediction])
    end
end
```

### Visualização de Mudanças

```julia
using Plots

# Features mudadas
changed_idx = result[:changed_indices]
x_before = x_factual[changed_idx]
x_after = result[:counterfactual][changed_idx]

# Plot
scatter(1:length(changed_idx), x_before, label="Before", marker=:circle)
scatter!(1:length(changed_idx), x_after, label="After", marker=:square)
xlabel!("Feature index (among changed)")
ylabel!("Feature value")
title!("Counterfactual Changes")
```

---

## Limitações

1. **Solver comercial necessário:** Gurobi é comercial (mas gratuito para acadêmicos)
2. **Escalabilidade:** MIPs podem ser lentos para redes grandes (>500 features)
3. **Precisão numérica:** ReLU epigraph pode ter pequenos erros numéricos
4. **Solução local:** Pode não encontrar o contrafactual de menor distância global

---

## Próximos Passos

1. ✅ Implementação básica concluída
2. 🔲 Testar em múltiplos sistemas (case118, case300, etc.)
3. 🔲 Comparar com solver direto DC-OPF
4. 🔲 Análise de robustez dos contrafactuais
5. 🔲 Incorporar constraints físicas adicionais
6. 🔲 Explorar counterfactuals "diverse" (múltiplas soluções)

---

**Autor:** Claude & Gabriel Medeiros
**Data:** 2025-10-29
