
PROJECT_ID="project_id"
TPU_ZONE="us-central1-b"
TPU_NAME="tpu-1"
LOCAL_SCRIPTS="/path/to/scripts"

# create a tpu-vm instance
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --zone "${TPU_ZONE}" \
  --accelerator-type "v2-8" \
  --version "tpu-vm-tf-2.11.0"       
  
# starts the tpu-vm instance
gcloud compute tpus tpu-vm start "${TPU_NAME}" \
  --zone "${TPU_ZONE}" --project "${PROJECT_ID}"

# stops the tpu-vm instance
gcloud compute tpus tpu-vm start "${TPU_NAME}" \
  --zone "${TPU_ZONE}" --project "${PROJECT_ID}"

# ssh into the tpu-vm instance
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --zone "${TPU_ZONE}" --project "${PROJECT_ID}"

# install miniconda on the tpu instance
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b
rm Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
./miniconda3/bin/conda init
# exit # re-ssh into tpu to check conda init

# install tpu compliant version tensorflow on the tpu instance
pip install --upgrade pip
pip install /usr/share/tpu/tensorflow-*.whl
# exit

# scp local files to the tpu-vm instance
gcloud compute tpus tpu-vm scp \
  "${LOCAL_SCRIPTS}:~/train_model" \
  --project "${PROJECT_ID}" \
  --zone "${tpu_zone}" \
  --recurse