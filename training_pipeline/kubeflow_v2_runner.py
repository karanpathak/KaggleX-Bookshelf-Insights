import pandas as pd
import requests
import os
import gzip
import json
import numpy as np

import pprint
import tempfile

from typing import Dict, Text

import tensorflow as tf
from tfx import v1 as tfx
import kfp
# from transformers import AutoTokenizer
# from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
# import tensorflow_text
# import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
# import tensorflow_datasets as tfds

from absl import logging

GOOGLE_CLOUD_PROJECT = 'kagglx-book-recommender'
GOOGLE_CLOUD_REGION = 'us-central1'
GCS_BUCKET_NAME = 'kagglx-book-recommender-bucket'

if not (GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_REGION and GCS_BUCKET_NAME):
    from absl import logging
    logging.error('Please set all required parameters.')

    

PIPELINE_NAME = 'book-recs-vertex-training'

# Path to various pipeline artifact.
PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/pipeline_module/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' data.
DATA_ROOT = 'gs://{}/data/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Name of Vertex AI Endpoint.
ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME

SERVING_MODEL_DIR = 'serving_model'

METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

print('PIPELINE_ROOT: {}'.format(PIPELINE_ROOT))

_trainer_module_file = 'trainer_module.py'
_books_transform_module_file = 'books_transform_module.py'
_ratings_transform_module_file = 'ratings_transform_module.py'

# def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
#                      module_file: str, endpoint_name: str, project_id: str,
#                      region: str, metadata_path: str, use_gpu: bool, module_root_arg: str) -> tfx.dsl.Pipeline:
def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, endpoint_name: str, project_id: str,
                     region: str, use_gpu: bool, module_root_arg: str) -> tfx.dsl.Pipeline:
    
    ratings_example_gen = tfx.components.CsvExampleGen(
        input_base=f'{data_root}/ratings/')
    
    books_example_gen = tfx.components.CsvExampleGen(
        input_base=f'{data_root}/books/')

    books_stats_gen = tfx.components.StatisticsGen(
        examples=books_example_gen.outputs['examples'])

    ratings_stats_gen = tfx.components.StatisticsGen(
        examples=ratings_example_gen.outputs['examples'])

    books_schema_gen = tfx.components.SchemaGen(
        statistics=books_stats_gen.outputs['statistics'],
        infer_feature_shape=False)

    ratings_schema_gen = tfx.components.SchemaGen(
        statistics=ratings_stats_gen.outputs['statistics'],
        infer_feature_shape=False)
    
    
    books_transform = tfx.components.Transform(
        examples=books_example_gen.outputs['examples'],
        schema=books_schema_gen.outputs['schema'],
        # module_file=os.path.abspath(_books_transform_module_file)
        module_file=os.path.join(module_root_arg, _books_transform_module_file)
    )

    ratings_transform = tfx.components.Transform(
        examples=ratings_example_gen.outputs['examples'],
        schema=ratings_schema_gen.outputs['schema'],
        # module_file=os.path.abspath(_ratings_transform_module_file)
        module_file=os.path.join(module_root_arg, _ratings_transform_module_file)
    )

    vertex_job_spec = {
          'project': project_id,
          'worker_pool_specs': [{
              'machine_spec': {
                  'machine_type': 'n1-standard-4',
              },
              'replica_count': 1,
              'container_spec': {
                  # 'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
                  'image_uri': 'gcr.io/kagglx-book-recommender/cb-tfx:latest'
              },
          }],
      }
    if use_gpu:
        # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
        # for available machine types.
        vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
    })
    
    # print("BOOKS URI COMPONENTS")
    # print(books_transform.outputs['transformed_examples'])
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
    # trainer = tfx.components.Trainer(
        # module_file=os.path.abspath(_trainer_module_file),
        module_file=module_file,
        examples=ratings_transform.outputs['transformed_examples'],
        transform_graph=ratings_transform.outputs['transform_graph'],
        schema=ratings_transform.outputs['post_transform_schema'],
        train_args=tfx.proto.TrainArgs(num_steps=50),
        eval_args=tfx.proto.EvalArgs(num_steps=10),
        custom_config={
            'epochs':2,
            'books':books_transform.outputs['transformed_examples'],
            'books_schema':books_transform.outputs['post_transform_schema'],
            'ratings':ratings_transform.outputs['transformed_examples'],
            'ratings_schema':ratings_transform.outputs['post_transform_schema'],
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:vertex_job_spec,
            'use_gpu':use_gpu,
        }
    )

    _serving_model_dir = 'gs://kagglx-book-recommender-bucket/models/candidate_model10'

    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=_serving_model_dir)))


    components = [
        ratings_example_gen.with_id('ratings_example_gen'),
        books_example_gen.with_id('books_example_gen'),
        books_stats_gen.with_id('books_stats_gen'),
        ratings_stats_gen.with_id('ratings_stats_gen'),
        books_schema_gen.with_id('books_schema_gen'),
        ratings_schema_gen.with_id('ratings_schema_gen'),
        books_transform.with_id('books_transform'),
        ratings_transform.with_id('ratings_transform'),
        trainer,
        pusher
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False)
    # return tfx.dsl.Pipeline(
    #   pipeline_name=pipeline_name,
    #   pipeline_root=pipeline_root,
    #   metadata_connection_config=tfx.orchestration.metadata
    #   .sqlite_metadata_connection_config(metadata_path),
    #   components=components)
    
    
def run():

    PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(default_image='gcr.io/kagglx-book-recommender/cb-tfx:latest'),
        output_filename=PIPELINE_DEFINITION_FILE)
    runner.run(
        _create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_root=DATA_ROOT,
            module_file=os.path.join(MODULE_ROOT, _trainer_module_file),
            endpoint_name=ENDPOINT_NAME,
            project_id=GOOGLE_CLOUD_PROJECT,
            region=GOOGLE_CLOUD_REGION,
            # We will use CPUs only for now.
            use_gpu=False,
            module_root_arg=MODULE_ROOT))

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()