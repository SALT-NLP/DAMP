export CUDA_VISIBLE_DEVICES=4,5

for DATASET in mtop top_v2 cstop_artificial
do
    python run_parsing.py \
	   --model_name_or_path google/mt5-small \
	   --do_train \
	   --do_eval \
	   --do_predict \
	   --learning_rate 1e-3 \
	   --weight_decay 0.0 \
	   --max_steps 3000 \
	   --evaluation_strategy "steps" \
	   --logging_steps 200 \
	   --eval_steps 200 \
	   --save_steps 200 \
	   --save_total_limit 1 \
	   --load_best_model_at_end True \
	   --dataset_name $DATASET \
	   --push_to_hub_model_id t5-small-pointer-$DATASET \
	   --push_to_hub True \
	   --output_dir /data/wheld3/mt5-small-pointer-$DATASET \
	   --per_device_train_batch_size=16 \
	   --gradient_accumulation_steps=32\
	   --per_device_eval_batch_size=16 \
	   --overwrite_output_dir \
	   --predict_with_generate \
	   --pointer_method
done
