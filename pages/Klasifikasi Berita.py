from function import *
import os
import joblib

st.header("Klasifikasi Artikel Berita Berdasarkan Hasil Summarization", divider='rainbow')

text = st.text_area("Masukkan Artikel Berita")
col1, col2 = st.columns(2)
with col1:
    k = st.number_input("Jumlah Kalimat Yang Di Ambil", 1)

with col2:
    threshold = st.number_input("Threshold")

button = st.button("Submit")

if "summary" not in st.session_state:
    st.session_state.summary = []
    st.session_state.graph_klasifikasi = []
    st.session_state.klasifikasi = ""

if button:
  text_clean = cleaning(text)
  tokenizing = tokenizer(text_clean)
  summary, G = summarization(x=tokenizing, k=k, threshold=threshold)
  
  st.session_state.summary = summary
  st.session_state.graph_klasifikasi = G


selected = option_menu(
  menu_title="",
  options=["Summary", "Klasifikasi", "Graph Kalimat"],
  icons=["data", "Process", "model", "implemen", "Test", "sa"],
  orientation="horizontal"
  )

if selected == "Summary":
  if st.session_state.summary:
    sentence = st.session_state.tokenSentence
    closeness = st.session_state.centrality
    st.write("**Hasil Summarization:**")
    st.write(st.session_state.summary)
    st.write("Nilai Closeness Centrality :")
    for i, cls in enumerate(closeness):
      st.write(f"index {i} Closeness : {closeness[cls]} -> Kalimat : {sentence[i]}")
  
  
elif selected == "Klasifikasi":
  if st.session_state.summary:
      st.caption("Klasifikasi Berdasarkan Hasil Summarization (Naive Bayes)")
      new_text = st.session_state.summary
      vectorizer = joblib.load("resources/vectorizer.pkl")
      nb = joblib.load("resources/modelNB.pkl")
      new_text_matrics = vectorizer.transform([new_text]).toarray()
      prediction = nb.predict(new_text_matrics)
      st.write("Prediction Category : ", prediction[0])
     

elif selected == "Graph Kalimat":
  col1, col2 = st.columns(2)
  with col1:
    x_canvas = st.number_input('Lebar Canvas', 5)
  
  with col2:
    y_canvas = st.number_input('Panjang Canvas', 5)
    
  node_size = st.number_input('Node Size', 400)
  
  if st.session_state.graph_klasifikasi != []:
    plot_graph(st.session_state.graph_klasifikasi, (x_canvas, y_canvas), node_size)