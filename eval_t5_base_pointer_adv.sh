export CUDA_VISIBLE_DEVICES=2,3

DATASET="mtop"
for LANG in en es fr de hi th
do
    python run_parsing.py \
	   --model_name_or_path WillHeld/t5-base-pointer-adv-$DATASET \
	   --do_predict \
	   --lang $LANG \
	   --output_dir /data/wheld3/mt5-base-pointer-adv-$DATASET/eval-$LANG \
	   --per_device_eval_batch_size=16 \
	   --dataset_name $DATASET \
	   --pointer_method \
	   --predict_with_generate
done

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-pointer-adv-top_v2\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-pointer-adv-top_v2/eval-hinglish \
       --per_device_eval_batch_size=16 \
       --dataset_name hinglish_top \
       --pointer_method \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-pointer-adv-top_v2\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-pointer-adv-top_v2/eval-en \
       --per_device_eval_batch_size=16 \
       --dataset_name top_v2 \
       --pointer_method \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-pointer-adv-cstop_artificial\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-pointer-adv-cstop_artificial/eval-spanglish \
       --per_device_eval_batch_size=32 \
       --dataset_name cstop \
       --pointer_method \
       --predict_with_generate

python run_parsing.py \
       --model_name_or_path WillHeld/t5-base-pointer-adv-cstop_artificial\
       --do_predict \
       --output_dir /data/wheld3/mt5-base-pointer-adv-cstop_artificial/eval-en \
       --per_device_eval_batch_size=32 \
       --dataset_name cstop_artificial \
       --pointer_method \
       --predict_with_generate
