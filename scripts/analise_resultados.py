# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Aluno:Dival Siqueira Neto
# RA: 801289
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics(model, best_params, best_score, auc_score, accuracy, f1, recall, logloss):
    """
    Esta função plota as métricas de desempenho para um modelo.

    Parâmetros:
    model (str): O nome do modelo.
    best_params (dict): Os melhores parâmetros encontrados para o modelo.
    best_score (float): A melhor pontuação obtida pelo modelo.
    auc_score (float): A pontuação AUC do modelo.
    accuracy (float): A acurácia do modelo.
    f1 (float): A pontuação F1 do modelo.
    recall (float): O recall do modelo.
    logloss (float): A log loss do modelo.
    """

    # Define as métricas e seus respectivos valores
    metrics = ['best_score', 'auc_score', 'accuracy', 'f1', 'recall', 'logloss']
    values = [best_score, auc_score, accuracy, f1, recall, logloss]

    # Cria um gráfico de barras das métricas
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color='skyblue')
    plt.xlabel('Métricas')
    plt.ylabel('Valores')
    plt.title(f'Métricas de Desempenho para o Modelo {model}')
    plt.show()
    
def plot_auc_roc(models, auc_scores):
    """
    Esta função plota o score AUC-ROC para cada modelo.
    
    Parâmetros:
    models (list): Uma lista de strings contendo os nomes dos modelos.
    auc_scores (list): Uma lista de floats contendo os scores AUC-ROC dos modelos.
    """
    # Cria uma nova figura com tamanho especificado
    plt.figure(figsize=(10, 5))
    # Cria um gráfico de barras com os nomes dos modelos no eixo x e os scores AUC-ROC no eixo y
    plt.bar(models, auc_scores, color=['blue', 'orange', 'green', 'red'])
    # Define o rótulo do eixo x
    plt.xlabel('Models')
    # Define o rótulo do eixo y
    plt.ylabel('AUC-ROC Score')
    # Define o título do gráfico
    plt.title('AUC-ROC Score of Models')
    # Exibe o gráfico
    plt.show()