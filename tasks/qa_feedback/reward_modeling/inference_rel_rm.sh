torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_fg_rm.py \
                --model_name_or_path /home/ec2-user/GREASE-AmbPrompts/assets/rel_rm \
                --train_file ./tasks/qa_feedback/data/NF-ERR_subsentence/train.json \
                --validation_file ./tasks/qa_feedback/data/NF-ERR_subsentence/dev.json \
                --test_file ./tasks/qa_feedback/data/NF-ERR_subsentence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/rel_rm \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 24 \
                --max_seq_length 2048 \