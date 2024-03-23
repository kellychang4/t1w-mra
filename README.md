# t1w-mra
Scripts for T1w to MRA Project

# Create TFRecord Dataset

The `tfrecord.py` script provides a method to create a TFRecord dataset (.tfrec) from a .csv file. This process should be repeated to create a `train`, `test`, and `valid` TFRecord dataset prior to model training. Below is an example command line call to create a TFRecord dataset for a train dataset.

```bash
python tfrecord.py \
  --file_prefix "train" \             # choices: train, test, valid (required)
  --csv_fname "train_split.csv" \     # path to csv, see below for format (required)
  --output_dir "/path/to/output" \    # path to output directory (required)
  --examples_per_shard 128 \          # number of image/label pairs in a TFRecord file (optional)
  --compression_type "GZIP"           # compression type on TFRecord file (optional)
```

The CSV file is expected to adhere to this format:

```text
image,label
/path/to/image_1.png,/path/to/label_1.png
/path/to/image_2.png,/path/to/label_2.png
/path/to/image_3.png,/path/to/label_3.png
<...>
```

The output directory will contain the TFRecord dataset shards.

```text
output_dir/
├── <file_prefix>_shard-000.tfrec
├── <file_prefix>_shard-001.tfrec
├── <file_prefix>_shard-002.tfrec
└── ...
```

# Google Cloud Storage Organization

The training scripts expect the Google Cloud Storage organization to follow:

```text
bucket_name/                  # Google cloud storage bucket
├── tfrec/                    # TFRecord dataset directory
│   ├── <dataset_name>        # specific TFRecord dataset
│   └── ...
└── jobs/                     # all training jobs output 
    ├── <job_name>            # specific job output directory 
    │   ├── checkpoints       # checkpoint models saved after each epoch
    │   ├── logs              # tensorboard logging directory
    │   ├── model             # final training model outputs
    │   └── <job_name>.csv    # model training csv logger
    └── ...
```

- A `dataset_name` directory should contain `train`, `test`, and `valid` shards as created with `tfrecord.py`. 
- The contents of each `job_name` directory is created and saved during the model training process.

# Preparing the TPU for Model Training 

The `prepare_tpu.sh` script contains the commands required to prepare the TPU for model training. The bash script was written to store the commands and was not intended to be run as a script. The following commands discussed in this section are found in `prepare_tpu.sh`.

## Install Google Cloud CLI 

In order to interact with the TPU VM instance, you will need to install the [Google Cloud CLI](https://cloud.google.com/sdk/gcloud). Download and installation instructions can be found [here](https://cloud.google.com/sdk/docs/install).

## Acquire a TPU Instance

Next, a TPU VM instance should be created. This can be done through either the Google Cloud Platform console under `Compute Engine > TPUs > Create TPU Node` or through the gcloud CLI. An example of creating a TPU with the CLI is provided below. 

```bash
gcloud compute tpus tpu-vm create "my-tpu-1" \     # tpu instance name
  --zone "us-central1-b" \                         # tpu zone, depends on accelerator-type
  --accelerator-type "v2-8" \                      # tpu system architecture type
  --version "tpu-vm-tf-2.11.0"                     # tpu tensorflow software version
```
- [List of TPU accelerator types by zone](https://cloud.google.com/tpu/docs/regions-zones#us)
- Ideally, the TPU zone is the same of the Google Cloud Storage bucket zone for I/O speed during model training.

## Start the TPU Instance

The TPU instance can be start with the Google Cloud Platform console or the gcloud CLI (below).

```bash
gcloud compute tpus tpu-vm start "my-tpu-1" \      # tpu instance name
  --zone "us-central1-b" \                         # tpu zone, depends on accelerator-type
  --project "project_id"                           # project id for the current invocation
```
- [Information about Google Cloud Projects](https://cloud.google.com/storage/docs/projects)
- Project information will become important to have the proper access permissions of the TFRecord dataset.

## Prepare the TPU Instance

We start preparing the TPU instance by SSH'ing into the VM instance.

```bash
gcloud compute tpus tpu-vm ssh "my-tpu-1" \      # tpu instance name
  --zone "us-central1-b" \                       # tpu zone, depends on accelerator-type
  --project "project_id"                         # project id for the current invocation
```

It is recommend to install Miniconda3 on the VM (Linux based). All Python packages should be installed within a Miniconda3 environment **except** TensorFlow.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b
rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
./miniconda3/bin/conda init
```

Install the TPU version of TensorFlow requested by the `--version` when the TPU was created. 

```bash
pip install --upgrade pip
pip install /usr/share/tpu/tensorflow-*.whl
```

After you have installed your desired packages you can logout of the TPU instance with `exit`. 

## Copy Scripts to TPU Instance

We can transfer our local scripts to the TPU instance with a SCP command.

```bash
gcloud compute tpus tpu-vm scp \
  "/path/to/scripts" \                  # local scripts directory
  "my-tpu-1:~/scripts" \                # tpu paths to transfer the directory
  --zone "us-central1-b" \              # tpu zone, depends on accelerator-type
  --project "project_id"                # project id for the current invocation
  --recurse                             # recursive copying
```

The model training scripts should all be in one directory on the TPU instance:

```text
scripts/
├── losses.py
├── models.py
├── train_model.py
└── utils.py
```

# Train the Model

From within the TPU instance, we can call on the `train_model.py` to perform the model training. An example command line call is provided below:

``` bash
python "train_model.py" \                    
  --image_model "UNet" \                               # image transfer model (choices: UNet, RUNet)
  --loss_function "PercetualLossVGG16" \               # loss function (choices: PercetualLossVGG16, PercetualLossVGG19)
  --loss_layer "block3_conv2" \                        # perceptual loss layer for evaluation (see note below)
  --gcs_bucket "bucket_name" \                         # Google Cloud Storage bucket name
  --dataset_name "dataset_name" \                      # TFRecord dataset name
  --job_name "job_name" \                              # model training job name, preferrably timestamped
  --batch_size 128 \                                   # training batch size, memory dependent
  --image_shape 100 100 \                              # image size in pixels, [height width]
  --n_train_images 100 \                               # number of image/label pairs in the training dataset
  --n_valid_images 20 \                                # number of image/label pairs in the validation dataset
  --n_epochs 200 \                                     # number of training epochs
  --tpu_specs "local" "us-central1-b" "project_id"     # tpu specifications, ["local", zone, project_id]
```
- The `--loss_layer` will depend on the specificed `--loss_function`. Inspect the TensorFlow Keras implementation of [VGG16](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16) and [VGG19](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg19/VGG19) for more information.
- See the expected Google Cloud Storage organization to understand the `--gcs_bucket`, `--dataset_name`, and `--batch_size` arguments.

# Stop the TPU Instance

To prevent unncessary charges, stop the TPU instance when it is not actively training. 

```bash
gcloud compute tpus tpu-vm stop "my-tpu-1"    # tpu instance name
```
