import tensorflow as tf
import tensorflow_transform as tft

import tensorflow_recommenders as tfrs

def preprocessing_fn(inputs):
  # We only want the book title
  return {'book_title': inputs['title']}