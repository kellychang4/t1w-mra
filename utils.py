import os
import os.path as op
import tensorflow as tf


def _tf_device_configuration(tpu_specs):
  """Configure TensorFlow to use TPUs or GPUs."""
  # Set environment variables for TPU/GPU performance
  os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
  try: # detect TPUs
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu     = tpu_specs["tpu"], 
      zone    = tpu_specs["zone"],
      project = tpu_specs["project"]
    )
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))
    strategy = tf.distribute.TPUStrategy(resolver)
  except ValueError: # detect GPUs
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    strategy = tf.distribute.MirroredStrategy()
  return strategy.scope()


def _define_gcs_paths(gcs_bucket, job_name, dataset_name):   
  """Define Google Cloud Storage paths for job artifacts."""
  # Define Google Cloud Storage Paths
  gcs_job   = f"gs://{gcs_bucket}/jobs/{job_name}"
  dirs_dict = {
    "data": f"gs://{gcs_bucket}/tfrec/{dataset_name}",        
    "tb":   op.join(gcs_job, "logs"), 
    "ckpt": op.join(gcs_job, "checkpoints"), 
    "csv":  op.join(gcs_job, f"{job_name}.csv"),
    "save": op.join(gcs_job, "model")
  }
  return dirs_dict 


def _define_callbacks(dirs_dict):
  """Define callbacks for training a model."""

  # Enable visualizations for TensorBoard
  tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir = dirs_dict["tb"] # log file directory
  )
  
  # Save model/model weights at checkpoint (end of each epoch)
  ckpt_fname = "epoch-{epoch:03d}_vloss-{val_loss:.2f}" # checkpoint file pattern
  model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath          = op.join(dirs_dict["ckpt"], ckpt_fname), # file pattern
    monitor           = "val_loss",  # monitored metric
    save_best_only    = False,       # save all checkpoints
    save_weights_only = False        # save entire model
  )

  # Streams epoch results to a csv filename
  csv_logger = tf.keras.callbacks.CSVLogger(
    filename = dirs_dict["csv"], # csv filename
    append   = True              # append to existing csv
  )
  
  # Stop training when a monitored metric has stopped improving
  early_stop = tf.keras.callbacks.EarlyStopping(
    monitor   = "val_loss", # monitored metric
    min_delta = 0.001,      # min change qualifies as improvement
    patience  = 20          # n_epochs threshold = no improvement
  )
  
  # Reduce learning rate when monitored metric has stopped improving
  reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor  = "val_loss", # monitored metric
    factor   = 0.5,        # lr reduction factor
    patience = 2,          # n_epochs threshold = no improvement
    verbose  = 1           # update message
  )
  
  # Return callbacks in order of execution priority
  return [tensorboard, model_ckpt, csv_logger, early_stop, reduce_lr]