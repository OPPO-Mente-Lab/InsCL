# arr: task order
# data_dir: train dataset path, modify the dataset path as needed. 
# root_model: the base model path, modify your cached model checkpoint path as needed.
arr=('Classification' 'Text_Quality_Evaluation' 'Code' 'Detection' 'Sentiment_Analysis' 'Comprehension' 'Closed_QA' 'Extraction' 'Dialogue' 'Program_Execution' 'Rewriting' 'Open_QA' 'Misc' 'Generation' 'Summarization' 'Mathematics')
data_dir='dataset/train'
root_model='llama_7B'
output_dir='curriculum_model'
nproc=8
train_batch_size=2
max_length=2048
epoch=2

for i in {0..15}
do
if [ $i == 0 ]; then
echo "Load data from ${data_dir}/${arr[i]}.jsonl"
echo "Start training ${arr[i]} on base model!"
torchrun --nproc_per_node=${nproc} --master_port=10034 train_FlashAttn.py \
    --model_name_or_path ${root_model} \
    --report_to none \
    --data_path ${data_dir}/${arr[i]}.jsonl \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${train_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --model_max_length ${max_length} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
continue
fi
echo "Load data from ${data_dir}/${arr[i]}.jsonl"
echo "Start training ${arr[i]} on ${arr[i-1]}_model!"
torchrun --nproc_per_node=${nproc} --master_port=10034 train_mem.py \
    --model_name_or_path ${output_dir}/${arr[i-1]} \
    --report_to none \
    --data_path ${data_dir}/${arr[i]}.jsonl \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${train_batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --model_max_length ${max_length} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
done