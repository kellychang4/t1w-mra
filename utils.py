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


def _define_paths(dataset_root, job_name, dataset_name):   
  """Define Google Cloud Storage paths for job artifacts."""
  # Define Google Cloud Storage Paths
  root_jobname   = f"{dataset_root}/jobs/{job_name}"
  dirs_dict = {
    "data": f"{dataset_root}/tfrec/",        
    "tb":   op.join(root_jobname, "logs"), 
    "ckpt": op.join(root_jobname, "checkpoints"), 
    "csv":  op.join(root_jobname, f"{job_name}.csv"),
    "save": op.join(root_jobname, "model")
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