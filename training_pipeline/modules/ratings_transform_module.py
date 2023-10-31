import tensorflow as tf
import tensorflow_transform as tft
import pdb

NUM_OOV_BUCKETS = 1

def preprocessing_fn(inputs):
    outputs = {}
    outputs['user_id'] = tft.sparse_tensor_to_dense_with_shape(inputs['user_id'], [None, 1], '-1')
    outputs['title'] = tft.sparse_tensor_to_dense_with_shape(inputs['title'], [None, 1], '-1')
    
    tft.compute_and_apply_vocabulary(
        inputs['user_id'],
        num_oov_buckets=NUM_OOV_BUCKETS,
        vocab_filename='user_id_vocab')
    
    tft.compute_and_apply_vocabulary(
        inputs['title'],
        num_oov_buckets=NUM_OOV_BUCKETS,
        vocab_filename='book_id_vocab')
    
    return outputs