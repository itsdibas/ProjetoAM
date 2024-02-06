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

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Baixar dados do NLTK, se não estiverem presentes
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Remover caracteres especiais e números
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    
    # Converter para minúsculas
    text = text.lower()
    
    # Remover pontuações adicionais
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenização usando NLTK
    words = nltk.word_tokenize(text)
    
    # Remover stop words
    # Existing stop words
    stop_words = set(stopwords.words('english'))

    # Additional stop words
    additional_stop_words = ['a', 'about', 'above', 'after', 'again', 'said', 'one', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

    # Combine the two lists
    stop_words = stop_words.union(additional_stop_words)
    
    # Now remove stop words
    words = [word for word in words if word not in stop_words]
    
    # Lematização
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Juntar as palavras novamente
    text = ' '.join(words)
    
    return text


def load_data(file_names, batch_size):
    # Inicializa uma lista vazia para armazenar os dataframes
    dfs = []

    # Itera sobre cada nome de arquivo na lista de nomes de arquivos
    for file_name in file_names:
        # Lê o arquivo CSV em partes (chunks) de tamanho batch_size
        for chunk in pd.read_csv(f"{file_name}.csv", chunksize=batch_size):
            # Verifica se a coluna 'content' existe no chunk
            if 'content' in chunk.columns:
                # Aplica a função preprocess_text a cada elemento da coluna 'content'
                chunk['content'] = chunk['content'].apply(preprocess_text)
                # Converte a coluna 'date' para o tipo datetime
                chunk['date'] = pd.to_datetime(chunk['date'])
                # Filtra o chunk para incluir apenas as linhas cujo ano é 2019 ou 2020
                chunk = chunk[chunk['date'].dt.year.isin([2019, 2020])]
                # Adiciona o chunk processado à lista dfs
                dfs.append(chunk)
            # Deleta o chunk para liberar memória
            del chunk

    # Se a lista dfs não estiver vazia, concatena todos os chunks em um único dataframe
    if dfs:
        news_data = pd.concat(dfs, ignore_index=True)

    # Retorna o dataframe
    return news_data

