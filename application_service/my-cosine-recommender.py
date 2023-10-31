import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import requests
from io import BytesIO
import pickle
import requests
import json
import plotly.graph_objects as go
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network

SERVER_URL = 'http://localhost:8501/v1/models/candidate_model_signature_def3:predict'

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Book Recommender")
st.text("Provide the name of a book")

# @st.cache(allow_output_mutation=True)
# def load_model():
#   model = tf.keras.models.load_model('./candidate_model4/')
#   return model

def get_predictions(node_list):
  all_similar = []
  for source, target in node_list:
    # input_data = tf.constant([target], dtype=tf.int64)
    # predictions = infer(input_data)['outputs'].numpy()
    predict_request = {"signature_name": "serving_default", "instances": [target]}

    data = json.dumps(predict_request)

    headers = {"content-type": "application/json"}

    response = requests.post(SERVER_URL, data=data, headers=headers)

    predictions = response.json()['predictions']

    books_similarity = cosine_similarity(books_embdeddings, np.array(predictions).reshape(-1,32))
    
    similar_books = list(enumerate(books_similarity))
    sorted_similar_books = sorted(similar_books, key=lambda x:x[1], reverse=True)

    top_ten_similar = [i for i, sim in sorted_similar_books[0:10]]

    edges = list(zip([top_ten_similar[0]]*10, top_ten_similar))
    all_similar.extend(edges)
    
  return all_similar

# with st.spinner('Loading Model Into Memory....'):
#   model = load_model()

def get_similaries(book_idx):
  predict_request = {"signature_name": "serving_default", "instances": [book_idx+1]}

  data = json.dumps(predict_request)

  headers = {"content-type": "application/json"}

  response = requests.post(SERVER_URL, data=data, headers=headers)

  predictions = response.json()['predictions']

  books_similarity = cosine_similarity(books_embdeddings, np.array(predictions).reshape(-1,32))
    
  similar_books = list(enumerate(books_similarity))
  sorted_similar_books = sorted(similar_books, key=lambda x:x[1], reverse=True)

  return sorted_similar_books[:10]





with open('candidate_model_embeddings.pkl', 'rb') as f:
    candidate_embedding_array = pickle.load(f)

with open('candidate_model_vocab.pkl', 'rb') as f:
    candidate_vocab_array = pickle.load(f)

books_embdeddings = candidate_embedding_array.numpy()
book_idx_name=[x.decode('utf-8') for x in candidate_vocab_array.tolist()]

title = st.text_input('Enter a Book Title.. ','The Hobbit')
# graph_depth = st.text_input('Enter a Graph Depth.. ', 2)
if title is not None:
    # content = requests.get(path).content

    st.write("Pulling recommendations :")
    with st.spinner('classifying.....'):
      # label =np.argmax(model.predict(decode_img(content)),axis=1)
      # depth = int(graph_depth)

      full_set = []

      idx = book_idx_name.index(title)

      top_similar = get_similaries(idx)

      print(top_similar)

      values = [sim for i, sim in top_similar[0:10]]
      values = np.concatenate(values)
      print(values)

      label_idx = [i for i, sim in top_similar[0:10]]

      full_set_labels = [book_idx_name[i-1] for i in label_idx]

      fig, ax = plt.subplots()

      # Example data
      # people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
      y_pos = np.arange(len(full_set_labels))
      # performance = 3 + 10 * np.random.rand(len(people))
      # error = np.random.rand(len(people))

      ax.barh(y_pos, values, align='center')
      ax.set_yticks(y_pos, labels=full_set_labels)
      ax.invert_yaxis()  # labels read top-to-bottom
      ax.set_xlabel('Cosine Similarity')
      ax.set_title('Similar Books')

      st.pyplot(fig)

      # print(full_set_labels)
      # data = pd.DataFrame({'Categories': full_set_labels,
      #                'Values': values})

      # # Create a bar chart using Altair
      # chart = alt.Chart(data).mark_bar().encode(
      #   x='Values',
      #   y='Categories'
      # )

      # st.altair_chart(chart, use_container_width=True)

      # fig, ax = plt.subplots()
      # ax.barh(full_set_labels, values)  # Swap categories and values
      # ax.set_xlabel('Values')
      # ax.set_ylabel('Categories')
      # ax.set_title('Horizontal Bar Chart')

      # trace = go.Bar(
      #   x=values,  # Use values for the horizontal bars
      #   y=full_set_labels,  # Categories for the y-axis
      #   orientation='h'  # Set the orientation to horizontal
      # )

      # layout = go.Layout(
      #   title='Books Similarities',
      #   xaxis=dict(title='Values'),
      #   yaxis=dict(title='Categories')

      # )

      # fig = go.Figure(data=[trace], layout=layout)

      # # st.plotly_chart(fig, use_container_width=True)
      # # st.write(fig.to_html())

      # components.html(fig.to_html())

      # st.write(fig)

      # top_similar = [(idx+1,idx+1)]

      # for i in range(depth):
      #   top_similar = get_predictions(top_similar)
      #   full_set.extend(top_similar)

      # book_set = list(set([target for source, target in full_set]))
      # full_set_labels = [book_idx_name[i-1] for i in book_set]


      # net = Network(notebook=True, cdn_resources='remote')

      # net.add_nodes(book_set,
      #         label=full_set_labels)

      # net.add_edges(full_set)

      # net.save_graph('top_ten_graphed.html')

      # HtmlFile = open(f'top_ten_graphed.html','r',encoding='utf-8')

      # components.html(HtmlFile.read(), height=500)
      # st.write(classes[label[0]])    
    # st.write("")
    # image = Image.open(BytesIO(content))
    # st.image(image, caption='Classifying Image', use_column_width=True)
    # st.write(', '.join(','.join(str(x) for x in embeddings.tolist()[0])))