from function import *

st.header("Ekstraksi Kata Kunci Artikel Berita", divider='rainbow')

num_word = int(st.number_input("Jumlah Kata Kunci", 1))
text = st.text_area("Masukkan Artikel Berita")
button = st.button("Ekstrak Kata Kunci")

if "keywoards" not in st.session_state:
    st.session_state.keywoards = []
    st.session_state.graph = []
    st.session_state.co_occurence = ""

if button:
  text_clean = cleaning(text)
  tokenizing = tokenizer(text_clean)
  G, keywoards = extract_top_words(x=tokenizing, w=num_word)
  
  # Menambahkan hasil keywoards ke dalam session state
  st.session_state.keywoards = keywoards
  st.session_state.graph = G


selected = option_menu(
  menu_title="",
  options=["Keywoards", "Graph Keywoards", "Co-occurence Matrics"],
  icons=["data", "Process", "model", "implemen", "Test", "sa"],
  orientation="horizontal"
  )

if selected == "Keywoards":
  if st.session_state.keywoards:
    st.write(st.session_state.keywoards)
  
elif selected == "Graph Keywoards":
  col1, col2 = st.columns(2)
  with col1:
    x_canvas = st.number_input('Lebar Canvas', 5)
  
  with col2:
    y_canvas = st.number_input('Panjang Canvas', 5)
    
  node_size = st.number_input('Node Size', 400)
  
  if st.session_state.graph != []:
    plot_graph(st.session_state.graph, (x_canvas, y_canvas), node_size)
  
elif selected == "Co-occurence Matrics":
  if str(type(st.session_state.co_occurence)) == "<class 'pandas.core.frame.DataFrame'>":
    st.dataframe(st.session_state.co_occurence, use_container_width=True)
