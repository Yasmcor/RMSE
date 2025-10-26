# 🐍 RMSE Calculator: Raiz do Erro Quadrático Médio

Este repositório contém um *notebook* Jupyter interativo (`rmse_calculator.ipynb`) para entender e calcular a métrica **RMSE (Root Mean Squared Error)**.

O **RMSE** é uma métrica fundamental em Machine Learning e Estatística para medir a magnitude média dos erros em previsões. Ele é especialmente útil porque penaliza mais os **erros grandes**, tornando-se ideal para modelos onde grandes desvios são mais custosos.

## 🎯 Por Que o RMSE?

A principal vantagem do RMSE é sua sensibilidade a *outliers* (erros grandes). Como os erros são elevados ao quadrado antes de serem somados e terem a média calculada, um erro de 10 é penalizado 100 vezes, enquanto dois erros de 5 são penalizados 50 vezes (25 + 25).

| Característica | Detalhe |
| :--- | :--- |
| **Definição** | Mede a diferença média entre os valores previstos por um modelo e os valores reais, na mesma unidade da variável de saída. |
| **Penalidade** | **Penaliza erros grandes** de forma desproporcional. |
| **Uso Ideal** | Previsão de séries temporais, modelos de regressão, e cenários onde o custo de um grande erro é muito alto (ex: controle de estoque, finanças). |

---

## 💻 Estrutura do Projeto

O projeto é composto por um único arquivo central:

* `rmse_calculator.ipynb`: O *notebook* Jupyter que contém a explicação detalhada do RMSE em português, inglês e francês, juntamente com a **função Python** e a **interface interativa (Widgets)** para cálculo.

### Como Usar

Para executar a calculadora interativa, você precisa ter o Jupyter Notebook (ou JupyterLab) instalado.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SeuUsuario/rmse-calculator.git](https://github.com/SeuUsuario/rmse-calculator.git)
    cd rmse-calculator
    ```

2.  **Instale as dependências:**
    ```bash
    pip install numpy ipywidgets jupyter
    ```
    *Obs.: A biblioteca `ipywidgets` é essencial para a interface interativa (input de dados).*

3.  **Execute o Notebook:**
    ```bash
    jupyter notebook rmse_calculator.ipynb
    ```

4.  **Interaja:** Abra o notebook no seu navegador e execute as células. Insira seus **valores reais** e **valores previstos** separados por vírgula na interface de *widgets* e clique em **"Calcular RMSE"** para ver o resultado instantaneamente.

---

## 🛠️ Detalhes da Implementação

A lógica central de cálculo é implementada na função `rmse(y_true, y_pred)`:

```python
import numpy as np

def rmse(y_true, y_pred):
    # 1. Converte para arrays numpy para operações vetorizadas
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # 2. Calcula o Erro (Diferença)
    errors = y_pred - y_true
    
    # 3. Calcula o Erro Quadrático Médio (MSE)
    #    errors**2 eleva cada erro ao quadrado.
    #    np.mean() calcula a média.
    mse = np.mean(errors**2) 
    
    # 4. Retorna a Raiz Quadrada (RMSE)
    return np.sqrt(mse)
