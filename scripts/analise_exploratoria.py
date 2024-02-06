# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Aluno: Dival Siqueira Neto
# RA: 801289
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

def make_predictions_and_save(model, tfidf_vectorizer, filtered_data, filename):
    # Carrega os dados de teste do arquivo CSV
    test_data = pd.read_csv("test.csv")
    # Mantém apenas a coluna 'id' dos dados de teste
    test_data = test_data[['id']]
    # Une os dados de teste com os dados filtrados com base no 'id'
    test_news_data = pd.merge(test_data, filtered_data, on='id', how='left')

    print("----------------------------------")

    # Preenche os valores nulos na coluna 'content' com uma string vazia
    test_news_data['content'] = test_news_data['content'].fillna('')
    # Transforma o conteúdo em um vetor TF-IDF
    X_test = tfidf_vectorizer.transform(test_news_data['content'])
    # Faz as previsões com o modelo
    test_predictions = model.predict_proba(X_test)[:, 1]

    # Adiciona as previsões ao DataFrame
    test_news_data['label'] = test_predictions
    print(f'O tamanho do DataFrame de submissão: {len(test_news_data)}')

    # Cria um DataFrame para submissão e salva em um arquivo CSV
    submission_df = pd.DataFrame({'id': test_news_data['id'], 'label': test_news_data['label'].fillna(0)})
    submission_df.to_csv(filename, index=False)
    
def plot_label_distribution(news_data):
    # Conta o número de cada label
    label_counts = news_data['label'].value_counts(dropna=False)

    # Cria um gráfico de barras da contagem de labels
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.index.astype(str), label_counts.values, color=['blue', 'green', 'red'])
    plt.xlabel('Label')
    plt.ylabel('Contagem')
    plt.title('Distribuição de Labels nos dados de treino')
    plt.show()
    
def plot_fake_news_by_month(news_data, train_data):
    """
    Esta função une os DataFrames news_data e train_data, e plota o número de notícias falsas por mês.

    Parâmetros:
    news_data (pandas.DataFrame): O DataFrame contendo os dados das notícias.
    train_data (pandas.DataFrame): O DataFrame contendo os dados de treinamento.
    """

    # Une os DataFrames news_data e train_data
    merged_data = pd.merge(news_data, train_data, on='id')

    # Converte a coluna 'date' para datetime
    merged_data['date'] = pd.to_datetime(merged_data['date'])

    # Filtra as notícias falsas
    fake_news = merged_data[merged_data['label'] == 0]

    # Agrupa por mês e conta o número de notícias falsas
    fake_news_by_month = fake_news.resample('M', on='date').count()['id']

    # Cria um gráfico de barras do número de notícias falsas por mês
    plt.figure(figsize=(10, 6))
    plt.bar(fake_news_by_month.index, fake_news_by_month.values, color='blue')
    plt.xlabel('Mês')
    plt.ylabel('Número de Notícias Falsas')
    plt.title('Número de Notícias Falsas por Mês')
    plt.show()
    
    
def plot_word_frequency(news_data, train_data):
    """
    Esta função plota as 10 palavras mais comuns em notícias verdadeiras e falsas.

    Parâmetros:
    news_data (pandas.DataFrame): O DataFrame contendo os dados das notícias.
    train_data (pandas.DataFrame): O DataFrame contendo os dados de treinamento.
    """

    # Une os DataFrames na coluna 'id'
    merged_data = pd.merge(news_data, train_data, on='id')

    stop_words = set(stopwords.words('english'))

    # Filtra as notícias verdadeiras e falsas
    true_news = merged_data[merged_data['label'] == 1]
    fake_news = merged_data[merged_data['label'] == 0]

    # Inicializa os contadores
    true_word_counts = Counter()
    fake_word_counts = Counter()

    # Tokeniza o texto, remove as stop words, e conta as palavras
    for text in true_news['content']:
        true_word_counts.update(word for word in word_tokenize(text) if word.isalpha() and word not in stop_words)
    for text in fake_news['content']:
        fake_word_counts.update(word for word in word_tokenize(text) if word.isalpha() and word not in stop_words)

    # Obtém as 10 palavras mais comuns
    true_common_words = true_word_counts.most_common(10)
    fake_common_words = fake_word_counts.most_common(10)

    # Cria um gráfico de barras das 10 palavras mais comuns em notícias verdadeiras
    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*true_common_words), color='blue')
    plt.xlabel('Palavras')
    plt.ylabel('Frequência')
    plt.title('10 Palavras Mais Comuns em Notícias Verdadeiras')
    plt.show()

    # Cria um gráfico de barras das 10 palavras mais comuns em notícias falsas
    plt.figure(figsize=(10, 6))
    plt.bar(*zip(*fake_common_words), color='red')
    plt.xlabel('Palavras')
    plt.ylabel('Frequência')
    plt.title('10 Palavras Mais Comuns em Notícias Falsas')
    plt.show()