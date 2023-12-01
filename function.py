from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
import requests 

nltk.download('stopwords')
nltk.download('punkt')

def cleaning(text):
  text = re.sub(r'[^\w\s.?!,]', '', text).strip().lower()
  return text

corpus = stopwords.words('indonesian')
def stopwords(text):
  return ' '.join(word for word in text.split() if word not in corpus)

def tokenizer(text):
  text = text.lower()
  return sent_tokenize(text)

def plot_graph(G, figsize=(35, 30), node_size=700, node_color='skyblue'):
  # Menggambar graf dengan canvas yang diperbesar
  pos = nx.spring_layout(G)  # Menentukan posisi simpul
  labels = nx.get_edge_attributes(G, 'weight')

  # Menentukan ukuran canvas
  fig = plt.figure(figsize=figsize)

  # Menggambar graf dengan ukuran canvas yang diperbesar
  nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color)
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
  # plt.show()
  st.pyplot(fig)
  

def graph_co_occurrence(x, threshold=0, show_matrics = False):
  vectorizer = CountVectorizer()
  tfidf_matrics = vectorizer.fit_transform(x)
  co_occurrence_matrix = tfidf_matrics.T.dot(tfidf_matrics).toarray()
  df_co_occurence = pd.DataFrame(co_occurrence_matrix, columns=vectorizer.get_feature_names_out())
  df_co_occurence.insert(0, 'Words', vectorizer.get_feature_names_out())
  
  st.session_state.co_occurence = df_co_occurence

  G = nx.DiGraph()

  # Menambahkan edge ke graf berdasarkan matriks co-occurrence
  for i in range(len(co_occurrence_matrix)):
    for j in range(i + 1, len(co_occurrence_matrix)):
      weight = co_occurrence_matrix[i, j]
      if weight > threshold:
        G.add_edge(vectorizer.get_feature_names_out()[i], vectorizer.get_feature_names_out()[j], weight=weight)
  return G

def extract_top_words(x, w=3, threshold=0, show_matrics=False, index=1):
    full_text = ' '.join(word for word in x)

    G = graph_co_occurrence(x, threshold, show_matrics)

    # Menghitung nilai dari PageRank (TextRank)
    scores = nx.pagerank(G)

    # Dictionary untuk menyimpan skor tertinggi setiap kata
    ranked_words_dict = {}

    for word in ' '.join(x).split():
        current_score = scores.get(word, 0)
        if word not in ranked_words_dict or current_score > ranked_words_dict[word]:
            ranked_words_dict[word] = current_score

    # Mengurutkan kata-kata berdasarkan skor tertinggi
    ranked_words = sorted(((score, word) for word, score in ranked_words_dict.items()), key=lambda x: (x[0], x[1]), reverse=True)

    # Memilih sejumlah w kata tertinggi
    selected_words = [word for _, word in sorted(ranked_words[:w], key=lambda x: x[1])] if w is not None else None

    # Menggabungkan kata-kata menjadi satu string terpisah dengan koma
    keywords = ', '.join(selected_words) if selected_words else ''
    
    st.session_state.ranked_words = ranked_words
    st.session_state.full_text = full_text

    return (G, keywords)

def graph_cosine_sim(x, threshold = 0.11):
  # TFIDF
  vectorizer = TfidfVectorizer()
  tfidf = vectorizer.fit_transform(x)
  cos_sim = cosine_similarity(tfidf)
  G = nx.Graph()

  # Mengisi nilai similarity antara kalimat ke dalam edges (Garis Penghubung)
  for i in range(len(x)):
    for j in range(i+1, len(x)):
      sim = cos_sim[i][j]
      if sim > threshold:
        G.add_edge(i, j, weight=sim)

  return G

def summarization(x, k = 4, threshold=0.11):
  st.session_state.tokenSentence = x
  
  # Memasukkan Nilai Cosine Similirity ke dalam Graph
  G = graph_cosine_sim(x, threshold)

  # Menghitung nilai dari closeness centrality
  centrality = nx.closeness_centrality(G)
  
  st.session_state.centrality = centrality

  # Menyusun Kalimat berdasarkan nilai Closeness Centrality tertinggi dan lebih dari treshold
  centrality = dict(sorted(centrality.items(), key=lambda item : item[1], reverse=True))

  summary_sentences = []
  for i, centr in enumerate(centrality.items()):
    if i < k:
      summary_sentences.append(x[centr[0]])

  return ('_' + ' '.join(summary_sentences)+ '_', G)

