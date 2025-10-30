# Input Convex Neural Networks (ICNN) para DC-OPF

Implementação Julia de Input Convex Neural Networks (ICNN) para aproximação de funções convexas em problemas de otimização DC-OPF (DC Optimal Power Flow) e geração de explicações contrafactuais.

---

## 📋 Visão Geral

Este projeto implementa uma arquitetura **simplificada de ICNN** para **regressão**, adequada para aprender a função convexa que mapeia demandas do sistema elétrico para custos operacionais ótimos.

### Principais Características

- ✅ **Arquitetura feed-forward pura** (x → y direto)
- ✅ **Garantia de convexidade** via restrições W_i ≥ 0
- ✅ **Geração de contrafactuais via MIP** (Mixed-Integer Programming)
- ✅ **Dataset customizado** para DC-OPF
- ✅ **Documentação completa** em português

---

## 🚀 Quick Start

### 1. Instalação

```bash
# Ative o ambiente Julia
julia --project=.

# Instale dependências
using Pkg; Pkg.instantiate()
```

### 2. Gerar Dados DC-OPF

```bash
julia src/data/Generate_DCOPF.jl
```

### 3. Treinar Modelo ICNN

```bash
julia examples/train_dcopf.jl
```

### 4. Gerar Contrafactuais

```bash
julia examples/generate_counterfactual.jl
```

---

## 📁 Estrutura do Projeto

```
src/
├── ICNN.jl                          # Módulo principal
├── models/ficnn.jl                  # Modelo FICNN simplificado
├── training/trainer.jl              # Funções de treinamento
├── data/
│   ├── Generate_DCOPF.jl           # Gerador de dados DC-OPF
│   └── dcopf_loader.jl             # Carregador de dados
└── utils/io.jl                      # Save/load modelos

counterfactuals/
├── model_loader.jl                  # Carregador de modelos treinados
└── algorithms/mip_counterfactual.jl # MIP para contrafactuais

examples/
├── train_dcopf.jl                  # Treinar ICNN
├── generate_counterfactual.jl      # Gerar contrafactuais
└── test_architecture.jl            # Testes da arquitetura
```

---

## 📚 Documentação

- **[ARCHITECTURE_CHANGES.md](ARCHITECTURE_CHANGES.md)** - Mudanças de arquitetura
- **[COUNTERFACTUALS_GUIDE.md](COUNTERFACTUALS_GUIDE.md)** - Guia de contrafactuais
- **[COUNTERFACTUALS_SUMMARY.md](COUNTERFACTUALS_SUMMARY.md)** - Resumo rápido

---

**Status:** ✅ Implementação completa

**Autores:** Gabriel Medeiros & Claude

**Data:** 2025-10-29
