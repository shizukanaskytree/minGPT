#export CUDA_VISIBLE_DEVICES=-1
python /home/wxf/minigpt_prj/minGPT/train_pile/train.py \
--per_device_train_batch_size 1 \
--dataloader_num_workers 1 \
--gradient_accumulation_steps 1 \
--output_dir temp \
--report_to none \
2>&1 | tee -a trainer_stderr.log