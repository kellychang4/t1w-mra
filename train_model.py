import argparse
import os.path as op
import tensorflow as tf

from utils import _define_gcs_paths
from utils import _define_callbacks
from utils import _tf_device_configuration

from losses import PerceptualLossVGG16, SSIMLoss
from models import UNet as IMAGE_TRANSFER_MODEL


def _read_tfrecord(file_pattern, batch_size, image_shape, n_epochs = None, 
                   cache = False, shuffle = False, shuffle_buffer_size = None, 
                   compression_type = "GZIP", num_parallel_calls = -1):
    
  if shuffle and not shuffle_buffer_size:
    raise Exception("If shuffling, must provide a numeric shuffle_buffer_size.")
  
  # define single tfrecord file reader (depends on compression_type)
  @tf.function
  def _read_tfrec(fname):
    return tf.data.TFRecordDataset(fname, compression_type = compression_type)
  
  # define parser function (depends on image_shape) with tf decorated
  @tf.function
  def _parse_tfrec(example):
    # parse tfrecord information from bytes
    features_parse_dict = {
        "image": tf.io.FixedLenFeature([], dtype = tf.string), 
        "label": tf.io.FixedLenFeature([], dtype = tf.string), 
    }
    example = tf.io.parse_single_example(example, features = features_parse_dict)    

    # decode from bytes and set dtype of image/label values
    example["image"] = tf.io.decode_png(example["image"], dtype = tf.uint8)
    example["label"] = tf.io.decode_png(example["label"], dtype = tf.uint8)

    # set image/label shape with grayscale channel last
    example["image"].set_shape([*image_shape, 1])
    example["label"].set_shape([*image_shape, 1])

    return ( example["image"], example["label"] )
  
  # define dataset preparation function 
  @tf.function
  def _prepare_dataset(image, label):
    # convert grayscale (1 channel) to rgb (3 channels)
    image = tf.image.grayscale_to_rgb(image)
    label = tf.image.grayscale_to_rgb(label)
    
    # coerce data to float within range [0, 1]
    image = tf.cast(image, dtype = tf.float32) / 255
    label = tf.cast(label, dtype = tf.float32) / 255
    return ( image, label )

  # ingest tfrecord files and define model input pipeline
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle = True)
  dataset = dataset.interleave(_read_tfrec, num_parallel_calls = num_parallel_calls)    
  dataset = dataset.map(_parse_tfrec, num_parallel_calls = num_parallel_calls)
  dataset = dataset.map(_prepare_dataset, num_parallel_calls = num_parallel_calls)
  dataset = dataset.repeat(n_epochs) # repeat entire dataset for n_epochs
  if cache: # if cache'ing dataset, either to storage or memory
    dataset = dataset.cache() # if possible, cache here
  if shuffle: # if shuffle invidual entries (not required for validation and test datasets)
    dataset = dataset.shuffle(buffer_size = shuffle_buffer_size) # shuffle entire dataset
  dataset = dataset.batch(batch_size = batch_size, drop_remainder = True) # batch images
  dataset = dataset.prefetch(buffer_size = num_parallel_calls) # prefetch dataset
  return dataset


def _read_and_preprocess_dataset(tfrec_dir, batch_size, image_shape, 
                                 n_images, n_epochs, num_parallel_calls):
  # Begin Dataset Loading Process
  dataset = dict(); steps_per_epoch = dict() # initialize
  for ds in ["train", "valid"]: # for each dataset type
    # Load Current Dataset Type
    dataset[ds] = _read_tfrecord(
      file_pattern        = op.join(tfrec_dir, f"{ds}_shard-*.tfrec"), 
      batch_size          = batch_size,    # number of examples to concurrently fit 
      image_shape         = image_shape,   # actual image shape
      n_epochs            = n_epochs[ds],  # number of epochs = repeats of dataset
      cache               = False,         # if cacheing dataset
      shuffle             = ds == "train", # shuffling examples if training
      shuffle_buffer_size = n_images[ds],  # if shuffling, n_images size buffer
      num_parallel_calls  = num_parallel_calls # number of parallel calls
    )

    # Calculate Steps in Dataset
    steps_per_epoch[ds] = n_images[ds] // batch_size
  return dataset, steps_per_epoch

    
def main(gcs_bucket, job_name, dataset_name, batch_size, image_shape, 
         n_train_images, n_valid_images, n_epochs, tpu_specs):   
    
  # Configure TensorFlow Optimization Seetings by Device Type
  scope = _tf_device_configuration(tpu_specs)
  
  # Define Local and GCS Directories from GCS
  dirs_dict = _define_gcs_paths(gcs_bucket, job_name, dataset_name)
  
  # Read and Preprocess Training, Validation, and Testing Datasets
  dataset, steps_per_epoch = _read_and_preprocess_dataset(
    tfrec_dir    = dirs_dict["data"], 
    batch_size   = batch_size,
    image_shape  = image_shape, 
    n_images     = { "train": n_train_images, "valid": n_valid_images },
    n_epochs     = { "train": n_epochs, "valid": 1 }, 
    num_parallel_calls = -1 # shortcut for tf.data.AUTOTUNE
  )

  # Compile Model within Training Strategy
  with scope: 
    # Define Optimizer with Initial Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)

    # Instantiate Model Architecture
    model = IMAGE_TRANSFER_MODEL(
      batch_size  = batch_size, 
      input_shape = [*image_shape, 3] # rgb  
    )

    # Define Loss Function
    loss = PerceptualLossVGG16(
      loss_layer  = "block2_conv2", 
      input_shape = [*image_shape, 3]
    )

    # Compile Model
    model.compile(
      optimizer = optimizer, 
      loss      = loss,  
      metrics   = [ SSIMLoss, tf.keras.metrics.RootMeanSquaredError() ]
    )
            
  # Define Callbacks
  callbacks = _define_callbacks(dirs_dict)
  
  # Train / Fit Model
  _ = model.fit(
    x                = dataset["train"], 
    batch_size       = batch_size, 
    epochs           = n_epochs, 
    steps_per_epoch  = steps_per_epoch["train"], 
    validation_data  = dataset["valid"], 
    validation_steps = steps_per_epoch["valid"], 
    validation_freq  = 1, 
    callbacks        = callbacks
  )

  # Save Trained Model Output
  model.save(dirs_dict["save"])


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gcs_bucket", type = str)
  parser.add_argument("--dataset_name", type = str)
  parser.add_argument("--job_name", type = str)
  parser.add_argument("--batch_size", type = int)
  parser.add_argument("--image_shape", type = int, nargs = 2)
  parser.add_argument("--n_train_images", type = int)
  parser.add_argument("--n_valid_images", type = int)
  parser.add_argument("--n_epochs", type = int)
  parser.add_argument("--tpu_specs", type = str, nargs = 3, 
                      default = [None, None, None]) 
  args = parser.parse_args()
  
  zipped_specs = zip(["tpu", "zone", "project"], args.tpu_specs)
  args.tpu_specs = {k: x for k, x in zipped_specs} # convert to dictionary
  
  argument_information = f"""
  Starting Model Training:
    -> Google Cloud Storage Bucket: {args.gcs_bucket}
    -> Dataset Name: {args.dataset_name}
    -> Job Name: {args.job_name}
    -> Batch Size: {args.batch_size}
    -> Image Shape: {args.image_shape}
    -> Training Volumes: {args.n_train_images}
    -> Validation Volumes: {args.n_valid_images}
    -> Epochs: {args.n_epochs}
    -> TPU Specs: {args.tpu_specs}
  """
  print(argument_information)

  main(
    gcs_bucket      = args.gcs_bucket, 
    dataset_name    = args.dataset_name,
    job_name        = args.job_name, 
    batch_size      = args.batch_size, 
    image_shape     = args.image_shape, 
    n_train_images  = args.n_train_images, 
    n_valid_images  = args.n_valid_images,
    n_epochs        = args.n_epochs,
    tpu_specs       = args.tpu_specs
  )