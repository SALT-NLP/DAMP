export CUDA_VISIBLE_DEVICES=5

DATASET="mtop"
for LANG in en es fr de hi th
do
    python run_parsing.py \
	   --model_name_or_path WillHeld/t5-base-vanilla-$DATASET \
	   --do_predict \
	   --lang $LANG \
	   --output_dir /data/wheld3/mt5-base-$DATASET/eval-$LANG \
	   --per_device_eval_batch_size=32 \
	   --dataset_name $DATASET \
	   --predict_with_generate
done

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-vanilla-top_v2\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-top_v2/eval-hinglish \
       --per_device_eval_batch_size=32 \
       --dataset_name hinglish_top \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-vanilla-top_v2\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-top_v2/eval-en \
       --per_device_eval_batch_size=32 \
       --dataset_name top_v2 \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-vanilla-cstop_artificial\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-cstop_artificial/eval-spanglish \
       --per_device_eval_batch_size=32 \
       --dataset_name cstop \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-vanilla-cstop_artificial\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-cstop_artificial/eval-en \
       --per_device_eval_batch_size=32 \
       --dataset_name cstop_artificial \
       --predict_with_generate
