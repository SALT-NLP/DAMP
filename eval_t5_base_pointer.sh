export CUDA_VISIBLE_DEVICES=6,7

DATASET="mtop"
for LANG in en es fr de hi th
do
    python run_parsing.py \
	   --model_name_or_path WillHeld/t5-base-pointer-$DATASET \
	   --do_predict \
	   --lang $LANG \
	   --output_dir /data/wheld3/mt5-base-pointer-$DATASET/eval-$LANG \
	   --per_device_eval_batch_size=16 \
	   --dataset_name $DATASET \
	   --pointer-method \
	   --predict_with_generate
done

python run_parsing.py \
       --model_name_or_path WillHeld/t5-small-pointer-top_v2\
       --do_predict \
       --output_dir /data/wheld3/mt5-small-pointer-top_v2/eval-hinglish \
       --per_device_eval_batch_size=32 \
       --dataset_name hinglish_top \
       --pointer-method \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-small-pointer-top_v2\
       --do_predict \
       --output_dir /data/wheld3/mt5-small-pointer-top_v2/eval-en \
       --per_device_eval_batch_size=32 \
       --dataset_name top_v2 \
       --pointer-method \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-small-pointer-cstop_artificial\
       --do_predict \
       --output_dir /data/wheld3/mt5-small-pointer-cstop_artificial/eval-spanglish \
       --per_device_eval_batch_size=32 \
       --dataset_name cstop \
       --pointer-method \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-small-pointer-cstop_artificial\
       --do_predict \
       --output_dir /data/wheld3/mt5-small-pointer-cstop_artificial/eval-en \
       --per_device_eval_batch_size=32 \
       --dataset_name cstop_artificial \
       --pointer-method \
       --predict_with_generate
