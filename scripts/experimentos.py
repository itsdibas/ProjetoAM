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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

# experiments.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from textblob import TextBlob
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, log_loss
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.preprocessing import LabelEncoder


def get_sentiment(text):
    """
    Esta função retorna a polaridade do sentimento de um texto.

    Parâmetros:
    text (str): O texto a ser analisado.

    Retorna:
    float: A polaridade do sentimento do texto. Se o texto não for uma string, retorna 0.0.
    """
    if isinstance(text, str):
        return TextBlob(text).sentiment.polarity
    else:
        return 0.0

def run_experiment_LogisticRegression(news_data, train_data):
    """
    Esta função executa um experimento de classificação usando a Regressão Logística.

    Parâmetros:
    news_data (DataFrame): O conjunto de dados de notícias.
    train_data (DataFrame): O conjunto de dados de treinamento.

    Retorna:
    model: O modelo treinado.
    best_params: Os melhores parâmetros encontrados para o modelo.
    best_score: A melhor pontuação obtida pelo modelo.
    auc_score: A pontuação AUC do modelo no conjunto de validação.
    accuracy: A acurácia do modelo no conjunto de validação.
    f1: A pontuação F1 do modelo no conjunto de validação.
    recall: O recall do modelo no conjunto de validação.
    logloss: A log loss do modelo no conjunto de validação.
    tfidf_vectorizer: O vetorizador TF-IDF usado para transformar o texto em características numéricas.
    """

    # Mescla os conjuntos de dados de notícias e treinamento
    merged_data = pd.merge(news_data, train_data[['id', 'label']], on='id', how='left')
    merged_data = merged_data.dropna(subset=['label'])

    # Calcula o comprimento do texto e o sentimento do conteúdo e do título
    merged_data['text_length'] = merged_data['content'].apply(len)
    merged_data['sentiment'] = merged_data['content'].apply(get_sentiment)
    merged_data['title_sentiment'] = merged_data['title'].apply(get_sentiment)

    # Converte a data para o formato datetime e ordena os dados por data
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    merged_data = merged_data.sort_values('date')

    # Divide os dados em conjuntos de treinamento e validação
    split_index = int(len(merged_data) * 0.8)
    train_set = merged_data[:split_index]
    val_set = merged_data[split_index:]

    # Transforma o texto em características numéricas usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=8000)
    X_train = tfidf_vectorizer.fit_transform(train_set['content'])
    y_train = train_set['label']

    # Treina o modelo de Regressão Logística com busca aleatória de hiperparâmetros
    model = LogisticRegression()
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    random_search.fit(X_train, y_train)

    # Imprime os melhores parâmetros e a melhor pontuação
    print(f'Best parameters: {random_search.best_params_}')
    print(f'Best score: {random_search.best_score_}')

    # Faz previsões no conjunto de validação
    model = random_search.best_estimator_
    X_val = tfidf_vectorizer.transform(val_set['content'])
    val_predictions_proba = model.predict_proba(X_val)[:, 1]
    val_predictions = model.predict(X_val)

    # Calcula as métricas de desempenho no conjunto de validação
    auc_score = roc_auc_score(val_set['label'], val_predictions_proba)
    accuracy = accuracy_score(val_set['label'], val_predictions)
    f1 = f1_score(val_set['label'], val_predictions)
    recall = recall_score(val_set['label'], val_predictions)
    logloss = log_loss(val_set['label'], val_predictions_proba)

    # Imprime as métricas de desempenho
    print(f'AUC-ROC Score on the validation set: {auc_score}')
    print(f'Accuracy on the validation set: {accuracy}')
    print(f'F1 Score on the validation set: {f1}')
    print(f'Recall on the validation set: {recall}')
    print(f'Log Loss on the validation set: {logloss}')

    # Retorna o modelo, os melhores parâmetros, a melhor pontuação e as métricas de desempenho
    return model, random_search.best_params_, random_search.best_score_, auc_score, accuracy, f1, recall, logloss, tfidf_vectorizer
            
def run_experiment_MultinomialNB(news_data, train_data):
    """
    Esta função executa um experimento de classificação usando o Naive Bayes Multinomial.

    Parâmetros:
    news_data (DataFrame): O conjunto de dados de notícias.
    train_data (DataFrame): O conjunto de dados de treinamento.

    Retorna:
    model: O modelo treinado.
    best_params: Os melhores parâmetros encontrados para o modelo.
    best_score: A melhor pontuação obtida pelo modelo.
    auc_score: A pontuação AUC do modelo no conjunto de validação.
    accuracy: A acurácia do modelo no conjunto de validação.
    f1: A pontuação F1 do modelo no conjunto de validação.
    recall: O recall do modelo no conjunto de validação.
    logloss: A log loss do modelo no conjunto de validação.
    tfidf_vectorizer: O vetorizador TF-IDF usado para transformar o texto em características numéricas.
    """

    # Mescla os conjuntos de dados de notícias e treinamento
    merged_data = pd.merge(news_data, train_data[['id', 'label']], on='id', how='left')
    merged_data = merged_data.dropna(subset=['label'])

    # Calcula o comprimento do texto e o sentimento do conteúdo e do título
    merged_data['text_length'] = merged_data['content'].apply(len)
    merged_data['sentiment'] = merged_data['content'].apply(get_sentiment)
    merged_data['title_sentiment'] = merged_data['title'].apply(get_sentiment)

    # Converte a data para o formato datetime e ordena os dados por data
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    merged_data = merged_data.sort_values('date')

    # Divide os dados em conjuntos de treinamento e validação
    split_index = int(len(merged_data) * 0.8)
    train_set = merged_data[:split_index]
    val_set = merged_data[split_index:]

    # Transforma o texto em características numéricas usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=8000)
    X_train = tfidf_vectorizer.fit_transform(train_set['content'])
    y_train = train_set['label']

    # Treina o modelo de Naive Bayes Multinomial com busca aleatória de hiperparâmetros
    model = MultinomialNB()
    param_grid = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='roc_auc', random_state=42)
    random_search.fit(X_train, y_train)

    # Imprime os melhores parâmetros e a melhor pontuação
    print(f'Best parameters: {random_search.best_params_}')
    print(f'Best score: {random_search.best_score_}')

    # Faz previsões no conjunto de validação
    model = random_search.best_estimator_
    X_val = tfidf_vectorizer.transform(val_set['content'])
    val_predictions_proba = model.predict_proba(X_val)[:, 1]
    val_predictions = model.predict(X_val)

    # Calcula as métricas de desempenho no conjunto de validação
    auc_score = roc_auc_score(val_set['label'], val_predictions_proba)
    accuracy = accuracy_score(val_set['label'], val_predictions)
    f1 = f1_score(val_set['label'], val_predictions)
    recall = recall_score(val_set['label'], val_predictions)
    logloss = log_loss(val_set['label'], val_predictions_proba)

    # Imprime as métricas de desempenho
    print(f'AUC-ROC Score on the validation set: {auc_score}')
    print(f'Accuracy on the validation set: {accuracy}')
    print(f'F1 Score on the validation set: {f1}')
    print(f'Recall on the validation set: {recall}')
    print(f'Log Loss on the validation set: {logloss}')

    # Retorna o modelo, os melhores parâmetros, a melhor pontuação e as métricas de desempenho
    return model, random_search.best_params_, random_search.best_score_, auc_score, accuracy, f1, recall, logloss, tfidf_vectorizer
            
            
            
def run_experiment_SVM(news_data, train_data):
    """
    Esta função executa um experimento de classificação usando o Support Vector Machine (SVM).

    Parâmetros:
    news_data (DataFrame): O conjunto de dados de notícias.
    train_data (DataFrame): O conjunto de dados de treinamento.

    Retorna:
    model: O modelo treinado.
    best_params: Os melhores parâmetros encontrados para o modelo.
    best_score: A melhor pontuação obtida pelo modelo.
    auc_score: A pontuação AUC do modelo no conjunto de validação.
    accuracy: A acurácia do modelo no conjunto de validação.
    f1: A pontuação F1 do modelo no conjunto de validação.
    recall: O recall do modelo no conjunto de validação.
    logloss: A log loss do modelo no conjunto de validação.
    tfidf_vectorizer: O vetorizador TF-IDF usado para transformar o texto em características numéricas.
    """

    # Mescla os conjuntos de dados de notícias e treinamento
    merged_data = pd.merge(news_data, train_data[['id', 'label']], on='id', how='left')
    merged_data = merged_data.dropna(subset=['label'])

    # Calcula o comprimento do texto e o sentimento do conteúdo e do título
    merged_data['text_length'] = merged_data['content'].apply(len)
    merged_data['sentiment'] = merged_data['content'].apply(get_sentiment)
    merged_data['title_sentiment'] = merged_data['title'].apply(get_sentiment)

    # Converte a data para o formato datetime e ordena os dados por data
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    merged_data = merged_data.sort_values('date')

    # Divide os dados em conjuntos de treinamento e validação
    split_index = int(len(merged_data) * 0.8)
    train_set = merged_data[:split_index]
    val_set = merged_data[split_index:]

    # Transforma o texto em características numéricas usando TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=30)
    X_train = tfidf_vectorizer.fit_transform(train_set['content'])
    y_train = train_set['label']

    # Treina o modelo SVM com busca aleatória de hiperparâmetros
    model = SVC(probability=True)
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']} 
    random_search = RandomizedSearchCV(model, param_grid, n_iter=1, cv=2, scoring='roc_auc', random_state=42)
    random_search.fit(X_train, y_train)

    # Imprime os melhores parâmetros e a melhor pontuação
    print(f'Best parameters: {random_search.best_params_}')
    print(f'Best score: {random_search.best_score_}')

    # Faz previsões no conjunto de validação
    model = random_search.best_estimator_
    X_val = tfidf_vectorizer.transform(val_set['content'])
    val_predictions_proba = model.predict_proba(X_val)[:, 1]
    val_predictions = model.predict(X_val)

    # Calcula as métricas de desempenho no conjunto de validação
    auc_score = roc_auc_score(val_set['label'], val_predictions_proba)
    accuracy = accuracy_score(val_set['label'], val_predictions)
    f1 = f1_score(val_set['label'], val_predictions)
    recall = recall_score(val_set['label'], val_predictions)
    logloss = log_loss(val_set['label'], val_predictions_proba)

    # Imprime as métricas de desempenho
    print(f'AUC-ROC Score on the validation set: {auc_score}')
    print(f'Accuracy on the validation set: {accuracy}')
    print(f'F1 Score on the validation set: {f1}')
    print(f'Recall on the validation set: {recall}')
    print(f'Log Loss on the validation set: {logloss}')

    # Retorna o modelo, os melhores parâmetros, a melhor pontuação e as métricas de desempenho
    return model, random_search.best_params_, random_search.best_score_, auc_score, accuracy, f1, recall, logloss, tfidf_vectorizer

def run_experiment_KNeighbors(news_data, train_data):
    # Merge the news and training datasets
    merged_data = pd.merge(news_data, train_data[['id', 'label']], on='id', how='left')
    merged_data = merged_data.dropna(subset=['label'])

    # Convert the date to datetime format and sort the data by date
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    merged_data = merged_data.sort_values('date')

    # Split the data into training and validation sets
    split_index = int(len(merged_data) * 0.8)
    train_set = merged_data[:split_index]
    val_set = merged_data[split_index:]

    # Transform the text into numerical features using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    X_train = tfidf_vectorizer.fit_transform(train_set['content'])
    y_train = train_set['label']

    # Train the KNeighbors model with random hyperparameter search
    model = KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(1, 31)), 'weights': ['uniform', 'distance']}
    random_search = RandomizedSearchCV(model, param_grid, n_iter=2, cv=2, scoring='roc_auc', random_state=42)
    random_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print(f'Best parameters: {random_search.best_params_}')
    print(f'Best score: {random_search.best_score_}')

    # Make predictions on the validation set
    model = random_search.best_estimator_
    X_val = tfidf_vectorizer.transform(val_set['content'])
    val_predictions_proba = model.predict_proba(X_val)[:, 1]
    val_predictions = model.predict(X_val)

    # Calculate the performance metrics on the validation set
    auc_score = roc_auc_score(val_set['label'], val_predictions_proba)
    accuracy = accuracy_score(val_set['label'], val_predictions)
    f1 = f1_score(val_set['label'], val_predictions)
    recall = recall_score(val_set['label'], val_predictions)
    logloss = log_loss(val_set['label'], val_predictions_proba)

    # Print the performance metrics
    print(f'AUC-ROC Score on the validation set: {auc_score}')
    print(f'Accuracy on the validation set: {accuracy}')
    print(f'F1 Score on the validation set: {f1}')
    print(f'Recall on the validation set: {recall}')
    print(f'Log Loss on the validation set: {logloss}')

    # Return the model, the best parameters, the best score, and the performance metrics
    return model, random_search.best_params_, random_search.best_score_, auc_score, accuracy, f1, recall, logloss, tfidf_vectorizer

