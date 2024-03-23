import os
import argparse
import numpy as np
import pandas as pd
import os.path as op
import tensorflow as tf          
  
  
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
      value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
  
  
def _tfrecord_example(image, label):
  """Creates a TFRecord Example, a paired image and label."""
  return tf.train.Example(features = tf.train.Features(feature = {
      "image": _bytes_feature(image), 
      "label": _bytes_feature(label) 
  }))


def _process_csv(csv_fname):
  """Processes image and label pairs from input csv file."""
  return [(x.image, x.label) for x in pd.read_csv(csv_fname).itertuples()]


def _write_tfrecord(example_list, shard_file_pattern,
                    examples_per_shard, compression_type):    
  """Writes a list of examples to TFRecords files (shards)."""
  
  # define tfrecord sharding parameters
  n_examples = len(example_list)
  n_shards   = np.ceil(n_examples / examples_per_shard).astype(np.int32)
  shard_list = np.array_split(example_list, n_shards)

  # write tfrecord shards to disk
  options = tf.io.TFRecordOptions(compression_type = compression_type)
  for n, shard in enumerate(shard_list): # for each shard
    shard_fname = shard_file_pattern.format(shard = n)
    writer = tf.io.TFRecordWriter(shard_fname, options = options) 
    for image, label in shard: # for each sample
      image = open(image, "rb").read() # load image as binary
      label = open(label, "rb").read() # load label as binary 
      example = _tfrecord_example(image, label) # create example
      writer.write(example.SerializeToString()) # write to shard
    writer.close() # close tfrec shard file
    print(f"  Saved {n+1:03d} out of {n_shards:03d} shards...")


def create_tfrecord_dataset(file_prefix, csv_fname, output_dir = os.getcwd(),        
                            examples_per_shard = 128, compression_type = "GZIP"):
  """Creates a TFRecord dataset from input CSV files."""

  # create local output directory (if does not exists)
  os.makedirs(output_dir, exist_ok = True)

  # create example list from csv file
  example_list = _process_csv(csv_fname)
      
  # write dataset tfrecord files
  print(f"Processing '{file_prefix}' dataset...")
  shard_name = f"{file_prefix}" + "_shard-{shard:03d}.tfrec"    
  _write_tfrecord(
    example_list       = example_list, 
    shard_file_pattern = op.join(output_dir, shard_name), 
    examples_per_shard = examples_per_shard, 
    compression_type   = compression_type
  )

      
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--file_prefix", type = str, required = True, 
                      choices = ["train", "test", "valid"])
  parser.add_argument("--csv_fname", type = str, required = True)
  parser.add_argument("--output_dir", type = str, default = os.getcwd())
  parser.add_argument("--examples_per_shard", type = int, default = 128)
  parser.add_argument("--compression_type", type = str, default = "GZIP")
  args = parser.parse_args()
          
  # print argument information
  print("\nStarting TFRecord Creation...")
  print(f"  -> File Prefix: {args.file_prefix}")
  print(f"  -> CSV Filename: {args.csv_fname}")
  print(f"  -> Output Directory: {args.output_dir}")
  print(f"  -> Examples per Shard: {args.examples_per_shard}")
  print(f"  -> Compression Type: {args.compression_type}\n")
  
  # create tfrecord dataset (with sharding)
  create_tfrecord_dataset(
    file_prefix        = args.file_prefix,
    csv_fname          = args.csv_fname, 
    output_dir         = args.output_dir,
    examples_per_shard = args.examples_per_shard, 
    compression_type   = args.compression_type
  )