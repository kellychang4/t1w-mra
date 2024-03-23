# t1w-mra
Scripts for T1w to MRA Project

# Create TFRecord Dataset

``` bash
python tfrecord.py \
  --train_csv "train_split.csv" \
  --test_csv "test_split.csv" \
  --valid_csv "valid_split.csv" \
  --output_dir "/path/to/output"
```

The structure of the CSV file should adhere to this format:

```
image,label
/path/to/image_1.png,/path/to/label_1.png
/path/to/image_2.png,/path/to/label_2.png
/path/to/image_3.png,/path/to/label_3.png
<...>
```

`train_split.csv`

# Utilities

Expected Google Cloud Storage organization:

`gs://{bucket_name}/tfrec/{dataset_name}`
`gs://{bucket_name}/job/{job_name}`
`gs://{bucket_name}/job/{job_name}/logs`
`gs://{bucket_name}/job/{job_name}/checkpoints`
`gs://{bucket_name}/job/{job_name}/model`
`gs://{bucket_name}/job/{job_name}/{job_name}.csv`

# Preparing the TPU for model training

``` bash
contents of prepare_tpu.sh script explained
```

# Train the Model

``` bash
python "train_model.py" \
  --image_model "UNet" \
  --loss_function "PercetualLossV166" \
  --loss_layer "block3_conv2" \
  --gcs_bucket "bucket_name" \
  --dataset_name "dataset_name" \
  --job_name "job_name" \
  --batch_size batch_size" \
  --image_shape 100 100 \
  --n_train_images "${n_train_images}" \
  --n_valid_images "${n_valid_images}" \
  --n_epochs 200 \
  --tpu_specs "local" "zone" "${project_id}"
```