### step. get the script file
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=${SCRIPT_DIR}/..
echo "current script name: $(basename $BASH_SOURCE), path: $SCRIPT_DIR, repo path: $REPO_DIR"

# used to store logs
mkdir -p ${SCRIPT_DIR}/tmp

export CUDA_VISIBLE_DEVICES=-1 # cpu only, use case: to get cpu memory usage of optimizer
python ${SCRIPT_DIR}/train.py \
--per_device_train_batch_size 1 \
--dataloader_num_workers 1 \
--gradient_accumulation_steps 1 \
--output_dir ${SCRIPT_DIR}/tmp/temp \
--report_to none \
2>&1 | tee -a ${SCRIPT_DIR}/tmp/trainer_stderr.log