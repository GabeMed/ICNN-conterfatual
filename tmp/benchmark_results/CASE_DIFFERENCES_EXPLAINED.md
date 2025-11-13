# O Que Varia Entre os Casos 1, 2 e 3?

## Resposta Curta

Cada caso representa um **cenário diferente** do problema DCOPF (DC Optimal Power Flow), com **configurações iniciais completamente diferentes** de geração e demanda de energia.

---

## Diferenças Específicas

### Caso 1
- **Test Index no dataset:** 3
- **Custo Factual (original):** 99,675
- **Características:**
  - Cenário com custo mais alto
  - Features originais específicas (ex: Feature 59 = 3.155, Feature 116 = 2.261)

### Caso 2  
- **Test Index no dataset:** 4
- **Custo Factual (original):** 95,440
- **Características:**
  - Cenário com custo médio-baixo
  - Features originais diferentes (ex: Feature 59 = 3.034, Feature 116 = 1.898)

### Caso 3
- **Test Index no dataset:** 12
- **Custo Factual (original):** 96,845
- **Características:**
  - Cenário com custo médio
  - Features originais únicas (ex: Feature 59 = 3.565, Feature 116 = 1.747)

---

## O Que Muda Exatamente?

### 1. **Vetor de Features Inicial (x_factual)**

Cada caso tem um vetor de 236 features diferentes representando:
- Níveis de geração de energia em diferentes geradores
- Demandas em diferentes nós da rede
- Condições operacionais da rede elétrica

**Exemplo - Feature 59 nos 3 casos:**
- Caso 1: 3.155
- Caso 2: 3.034
- Caso 3: 3.565

### 2. **Custo Original (y_factual)**

Resultado do modelo ICNN aplicado ao x_factual:
- Caso 1: 99,675
- Caso 2: 95,440  
- Caso 3: 96,845

### 3. **Dificuldade do Problema**

Alguns cenários são mais fáceis de otimizar:

**Para 20% de redução:**
- Caso 1: 5 features alteradas
- Caso 2: 5 features alteradas
- Caso 3: 4 features alteradas (mais fácil!)

**Para 30% de redução:**
- Caso 1: 9 features alteradas
- Caso 2: 8 features alteradas  
- Caso 3: 8 features alteradas

---

## Por Que Isso Importa?

### 1. **Validação da Robustez**

Testar múltiplos casos garante que os métodos funcionam em diferentes condições, não apenas em um cenário específico.

### 2. **Diferentes Dificuldades**

- Alguns casos precisam de mais mudanças para atingir a mesma % de redução
- Tempos de solução variam entre casos
- Mostra a generalização dos métodos

### 3. **Realismo**

Na prática, você encontrará diferentes configurações de rede. Os 3 casos simulam essa variabilidade.

---

## Analogia Prática

Imagine que você quer reduzir custos em 3 fábricas diferentes:

- **Fábrica 1 (Caso 1):** Custo alto (99,675), configuração complexa
- **Fábrica 2 (Caso 2):** Custo médio (95,440), configuração diferente  
- **Fábrica 3 (Caso 3):** Custo médio (96,845), outra configuração

Cada uma precisa de **intervenções diferentes** para atingir a mesma % de redução, mesmo que os métodos de otimização sejam os mesmos.

---

## Resumo Visual

```
CASO 1 (Test #3)
├─ Custo Original: 99,675 (ALTO)
├─ Features Iniciais: Configuração A
├─ Dificuldade: Média
└─ Exemplo: Feature 59 = 3.155

CASO 2 (Test #4)
├─ Custo Original: 95,440 (BAIXO)
├─ Features Iniciais: Configuração B
├─ Dificuldade: Baixa
└─ Exemplo: Feature 59 = 3.034

CASO 3 (Test #12)
├─ Custo Original: 96,845 (MÉDIO)
├─ Features Iniciais: Configuração C
├─ Dificuldade: Variável
└─ Exemplo: Feature 59 = 3.565
```

---

## Conclusão

**Os casos variam em:**
✓ Configuração inicial (236 features diferentes)  
✓ Custo original  
✓ Dificuldade de otimização  
✓ Número de features que precisam mudar

**Mas todos testam:**
✓ Os mesmos métodos (ECP, ESH, MILP)  
✓ As mesmas % de redução (20%, 30%, 40%, 50%, 60%)  
✓ As mesmas restrições (Target ± ε)

Isso garante que os resultados são **robustos** e **generalizáveis**!

