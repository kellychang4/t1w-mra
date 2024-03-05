# t1w-mra
Scripts for T1w to MRA Project

# Create TFRecord Dataset


```
python tfrecord.py \
  --train_csv "train_split.csv" \
  --test_csv "test_split.csv" \
  --valid_csv "valid_split.csv" \
  --output_dir "/path/to/output"
```

`train_split.csv`


# GSC paths 

`gs://{bucket_name}/tfrec/{dataset_name}`
`gs://{bucket_name}/job/{job_name}`
`gs://{bucket_name}/job/{job_name}/logs`
`gs://{bucket_name}/job/{job_name}/checkpoints`
`gs://{bucket_name}/job/{job_name}/model`
`gs://{bucket_name}/job/{job_name}/{job_name}.csv`