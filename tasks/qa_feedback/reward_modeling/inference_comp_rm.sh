# inference for getting mean std of COMP
torchrun --nproc_per_node 1 --standalone --nnodes=1 ./reward_modeling/run_pref_rm.py \
                --model_name_or_path /home/ec2-user/GREASE-AmbPrompts/assets/comp_rm \
                --validation_file ./tasks/qa_feedback/data/COMP_sequence/dev.json \
                --test_file ./tasks/qa_feedback/data/COMP_sequence/dev.json \
                --output_dir ./tasks/qa_feedback/model_outputs/comp_rm \
                --do_predict \
                --bf16 \
                --per_device_eval_batch_size 128 \
                --max_seq_length 2048 \
                --remove_unused_columns False \
                --cal_score_mean_std True
