
from typing import Dict, Text
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow_metadata.proto.v0 import schema_pb2
import tensorflow_recommenders as tfrs
from tensorflow_transform.tf_metadata import schema_utils
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

from typing import Dict, List, Text

import pdb

import os
import absl
import datetime
import glob
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_recommenders as tfrs

from absl import logging
from tfx.types import artifact_utils

from tfx import v1 as tfx
from tfx_bsl.coders import example_coder
from tfx_bsl.public import tfxio

import pickle

INPUT_FN_BATCH_SIZE = 1

from google.cloud import storage

def write_to_storage(data, file_name):
    client = storage.Client()
    bucket_name = 'kagglx-book-recommender-bucket'
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(data)

# def my_cloud_function(request):
#     data = 'Hello, world!'
#     file_name = 'hello.txt'
#     write_to_storage(data, file_name)
#     return 'File written to Google Cloud Storage!'


def extract_str_feature(dataset, feature_name):
    np_dataset = []
    for example in dataset:
        np_example = example_coder.ExampleToNumpyDict(example.numpy())
        np_dataset.append(np_example[feature_name][0].decode())
    return tf.data.Dataset.from_tensor_slices(np_dataset)

class UserModel(tf.keras.Model):
    def __init__(self, tf_transform_output):
        super().__init__()
        # unique_user_ids = tf_transform_output.vocabulary_by_name('user_id_vocab')
        # users_vocab_str = [b.decode() for b in unique_user_ids]
        # users_vocab_str = np.unique(unique_user_ids)
        ratings_df = pd.read_csv('gs://kagglx-book-recommender-bucket/data/book-recs-vertex-training/ratings/ratings_w_title.csv', names=['book_id', 'user_id', 'title'])
        ratings_df = ratings_df[['user_id', 'title']]
        ratings = tf.data.Dataset.from_tensor_slices((dict(ratings_df)))
        ratings = ratings.map(lambda x: {
            "title": x["title"],
            "user_id": x["user_id"],
        })
        
        users_vocab_str = np.unique(np.concatenate(list(ratings.batch(1_000).map(
            lambda x: x["user_id"]))))
        
        self.user_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=users_vocab_str, mask_token=None),
            tf.keras.layers.Embedding(len(users_vocab_str) + 1, 32)
        ])
        
    def call(self, inputs):
        return tf.concat([
            self.user_id_embedding(inputs),
        ], axis=1)

    
class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes, tf_transform_output):
        super().__init__()
        
        self.embedding_model = UserModel(tf_transform_output)
        
        self.dense_layers = tf.keras.Sequential()
        
        
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
            
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))
            
    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        
        return self.dense_layers(feature_embedding)

class BookModel(tf.keras.Model):
    
    def __init__(self, books_uri, tf_transform_output):
        
        super().__init__()
        print("BOOKS URI")
        print(books_uri)
#         '{data_root}/books/'
        
        # books_artifact = books_uri.get()[0]
        # input_dir = artifact_utils.get_split_uri([books_artifact], 'train')
        # book_files = glob.glob(os.path.join(input_dir, '*'))
        # books = tf.data.TFRecordDataset(book_files, compression_type="GZIP")
        # books = books.map(lambda x: x)
        
        books_df = pd.read_csv('gs://kagglx-book-recommender-bucket/data/book-recs-vertex-training/books/book_title.csv', names=['title'])
        books_df = books_df['title'].unique()
        books_df = pd.DataFrame(books_df, columns =['title'])
        books = tf.data.Dataset.from_tensor_slices((dict(books_df)))
        books = books.map(lambda x: x["title"])
        
        max_tokens = 10_000
        
        # unique_book_titles = tf_transform_output.vocabulary_by_name('book_id_vocab')
        # titles_vocab_str = [b.decode() for b in unique_book_titles]
        # titles_vocab_str = np.unique(unique_book_titles)
        titles_vocab_str = np.unique(np.concatenate(list(books.batch(1000))))
        
        # with open('gs://kagglx-book-recommender-bucket/vocabularies/candidate_model_vocab.pkl', 'wb') as f:
        #     pickle.dump(titles_vocab_str, f)
        
        pickled_titles = pickle.dumps(titles_vocab_str)
        write_to_storage(pickled_titles, 'vocabularies/candidate_model_vocab.pkl')
        
        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=titles_vocab_str,mask_token=None),
            tf.keras.layers.Embedding(len(titles_vocab_str) + 1, 32)
        ])
        
        self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        
        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])
        
        self.title_vectorizer.adapt(books)
        
    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)


class CandidateModel(tf.keras.Model):
    """Model for encoding books."""
    
    def __init__(self, layer_sizes, books_uri, tf_transform_output):
        """Model for encoding books.
        Args:
        layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains."""
        super().__init__()
        
        self.embedding_model = BookModel(books_uri, tf_transform_output)
        self.dense_layers = tf.keras.Sequential()
        
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
            
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))
            
    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class GoodreadsModel(tfrs.models.Model):
    def __init__(self, layer_sizes, books_uri, tf_transform_output):
        super().__init__()
        self.query_model = QueryModel(layer_sizes, tf_transform_output)
        self.candidate_model = CandidateModel(layer_sizes, books_uri, tf_transform_output)
        
        # books_artifact = books_uri.get()[0]
        # input_dir = artifact_utils.get_split_uri([books_artifact], 'train')
        # book_files = glob.glob(os.path.join(input_dir, '*'))
        # books = tf.data.TFRecordDataset(book_files, compression_type="GZIP")
        # books = books.map(lambda x: x)
        
        books_df = pd.read_csv('gs://kagglx-book-recommender-bucket/data/book-recs-vertex-training/books/book_title.csv', names=['title'])
        books_df = books_df['title'].unique()
        books_df = pd.DataFrame(books_df, columns =['title'])
        books = tf.data.Dataset.from_tensor_slices((dict(books_df)))
        books = books.map(lambda x: x["title"])
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
            candidates=books.batch(128).map(self.candidate_model),
            ),
        )
        
    def compute_loss(self, features, training=False):
        query_embeddings = tf.squeeze(self.query_model(features["user_id"]), axis=1)
        book_embeddings = self.candidate_model(features["title"])
        
        return self.task(query_embeddings, book_embeddings, compute_metrics=not training)

@tf.function
def serve_fn(input_data):
    layer_index = input_data[0]
    output_data = model.candidate_model.embedding_model.weights[0][layer_index]
    return {'outputs': output_data}

# This function will apply the same transform operation to training data
# and serving requests.
def _apply_preprocessing(raw_features, tft_layer):
    try:
        transformed_features = tft_layer(raw_features)
    except BaseException as err:
        logging.error('######## ERROR IN _apply_preprocessing:\n{}\n###############'.format(err))
    return transformed_features

def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    
    """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
    try:
        return data_accessor.tf_dataset_factory(
            file_pattern,
            tfxio.TensorFlowDatasetOptions(
                batch_size=batch_size),
            tf_transform_output.transformed_metadata.schema)
    except BaseException as err:
        logging.error('######## ERROR IN _input_fn:\n{}\n###############'.format(err))
        
    return None


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    """Train the model based on given args.
    
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                            tf_transform_output, INPUT_FN_BATCH_SIZE)
    
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                          tf_transform_output, INPUT_FN_BATCH_SIZE)
    
    model = GoodreadsModel([64, 32], fn_args.custom_config['books'], tf_transform_output)

  # tensorboard_callback = tf.keras.callbacks.TensorBoard(
  #       log_dir=fn_args.model_run_dir, update_freq='batch')
    
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    model.fit(
        train_dataset,
        epochs=fn_args.custom_config['epochs'],
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)
    
    # with open('gs://kagglx-book-recommender-bucket/vocabularies/candidate_model_embeddings.pkl', 'wb') as f:
    #     pickle.dump(model.candidate_model.embedding_model.weights[0], f)
    pickled_embeddings = pickle.dumps(model.candidate_model.embedding_model.weights[0])
    write_to_storage(pickled_embeddings, 'vocabularies/candidate_model_embeddings.pkl')

    # model.candidate_model.save(fn_args.serving_model_dir)
    # model.candidate_model.save('gs://kagglx-book-recommender-bucket/models/candidate_model9', save_format='tf')
    module = tf.Module()
    module.model = model.candidate_model
    module.serve = serve_fn
    
#     tf.saved_model.save(module,
#                       fn_args.serving_model_dir,
#                       signatures=serve_fn.get_concrete_function(
#                       input_data=tf.TensorSpec(shape=(None,), dtype=tf.int64)
#                       ),
#                       options=None
#                     )
    
    model.candidate_model.save(fn_args.serving_model_dir)