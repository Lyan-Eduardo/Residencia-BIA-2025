```markdown
# 🧠 Predictive Business Process Monitoring with LSTM (Python 3 Adaptation)

Este repositório é uma versão atualizada do projeto original de **Ilya Verenich** e **Niek Tax**, disponível em  
[github.com/verenich/ProcessSequencePrediction](https://github.com/verenich/ProcessSequencePrediction),  
baseado no artigo **"Predictive Business Process Monitoring with LSTM Neural Networks"**  
de *Niek Tax, Ilya Verenich, Marcello La Rosa e Marlon Dumas (CAiSE 2017)*.

---

## 🎓 Contexto acadêmico

Este projeto foi desenvolvido como parte do **Trabalho de Conclusão de Curso (TCC)** de  
**Lyan Eduardo Sakuno Rodrigues**, no curso de **Bacharelado em Inteligência Artificial**  
da **Universidade Federal de Goiás (UFG)**.

O objetivo é compreender e aplicar técnicas de **Process Mining** e **Deep Learning**  
para prever eventos futuros e tempos de execução em processos de negócio reais.

---

## ⚙️ Funcionalidades

A partir do código original, esta versão em **Python 3** permite realizar:

- 🔹 Predição da **próxima atividade** a ser executada em um processo em andamento;  
- 🔹 Predição do **timestamp da próxima atividade**;  
- 🔹 Predição da **continuação (sufixo)** de um processo em execução;  
- 🔹 Predição do **tempo restante total** de um caso.

O código foi atualizado para compatibilidade com **Keras 2.x / TensorFlow 2.x**  
e é totalmente funcional em ambientes locais (VS Code, PyCharm, terminal, etc.).

---

## 🧩 Estrutura dos scripts

| Script | Função Principal |
|--------|------------------|
| `Train.py` | Lê o log de eventos e treina um modelo LSTM para prever próxima atividade e tempo |
| `evaluate_suffix_and_remaining_time.py` | Avalia o modelo prevendo o sufixo e o tempo restante |
| `evaluate_next_activity_and_time.py` | Avalia o modelo na previsão da próxima atividade e tempo até ela |
| `calculate_accuracy_on_next_event.py` | Calcula a acurácia da predição da próxima atividade |

---

## 📂 Estrutura de pastas recomendada

```

projeto/
├── data/
│   └── helpdesk.csv
├── output_files/
│   ├── models/
│   └── results/
│       └── folds/
├── Train.py
├── evaluate_suffix_and_remaining_time.py
├── evaluate_next_activity_and_time.py
└── calculate_accuracy_on_next_event.py

````

---

## ⚙️ Configuração e execução local

### 🔧 Requisitos

Certifique-se de ter instalado:

- **Python 3.8+**
- **pip** atualizado
- **VS Code** (ou outro editor de sua preferência)

### 📦 Instalação das dependências

Execute no terminal do VS Code (ou CMD / PowerShell):

```bash
pip install numpy keras tensorflow scikit-learn distance jellyfish matplotlib
````

### 🧭 Passos de execução

1. **Treinar o modelo**

   ```bash
   python Train.py
   ```

   * Lê o arquivo `data/helpdesk.csv`
   * Gera modelos salvos em `output_files/models/`

2. **Avaliar sufixo e tempo restante**

   ```bash
   python evaluate_suffix_and_remaining_time.py
   ```

   * Carrega o modelo `.h5` salvo anteriormente
   * Gera resultados em `output_files/results/suffix_and_remaining_time_helpdesk.csv`

3. **Avaliar próxima atividade e tempo**

   ```bash
   python evaluate_next_activity_and_time.py
   ```

   * Gera o arquivo `output_files/results/next_activity_and_time_helpdesk.csv`

4. **Calcular acurácia da próxima atividade**

   ```bash
   python calculate_accuracy_on_next_event.py
   ```

   * Lê os resultados anteriores e calcula a acurácia por caso e total.

---

## 💾 Variáveis configuráveis

Nos scripts, as principais variáveis que podem ser alteradas são:

| Variável                  | Descrição                                                        |
| ------------------------- | ---------------------------------------------------------------- |
| `eventlog`                | Nome do arquivo de log (em `data/`)                              |
| `model = load_model(...)` | Caminho do modelo `.h5` a ser carregado                          |
| `predict_size`            | Quantidade de eventos futuros a prever (`1` = próxima atividade) |

---

## 🧠 Observações

* Os logs de eventos devem seguir o formato:

  ```
  CaseID,ActivityID,CompleteTimestamp
  1,12,2014-01-02 08:30:00
  1,15,2014-01-02 09:10:00
  2,7,2014-01-03 10:00:00
  ```
* O arquivo `helpdesk.csv` pode ser substituído por outros datasets (como *BPI Challenge* ou *Sepsis*), bastando ajustar o nome na variável `eventlog`.

---

## 📚 Referências

### 🔹 Artigo base

> **Predictive Business Process Monitoring with LSTM Neural Networks**
> Niek Tax, Ilya Verenich, Marcello La Rosa, and Marlon Dumas.
> *Proceedings of the 29th International Conference on Advanced Information Systems Engineering (CAiSE 2017)*.
> Springer, pp. 477–492.

```bibtex
@inproceedings{Tax2017,
  title     = {Predictive Business Process Monitoring with {LSTM} Neural Networks},
  author    = {Tax, Niek and Verenich, Ilya and La Rosa, Marcello and Dumas, Marlon},
  booktitle = {Proceedings of the 29th International Conference on Advanced Information Systems Engineering},
  year      = {2017},
  pages     = {477--492},
  publisher = {Springer}
}
```

### 🔹 Repositório original

* [github.com/verenich/ProcessSequencePrediction](https://github.com/verenich/ProcessSequencePrediction)

---

## 👤 Autor da adaptação

**Lyan Eduardo Sakuno Rodrigues**
Bacharelado em Inteligência Artificial – Universidade Federal de Goiás (UFG)
📧 [lyanesr@gmail.com](mailto:lyanesr@gmail.com)

---

## 📜 Licença

Esta é uma versão adaptada para fins acadêmicos e de pesquisa do trabalho original de **Ilya Verenich** e **Niek Tax**, com atualização completa para **Python 3** e compatibilidade com **TensorFlow/Keras 2.x**.

```
