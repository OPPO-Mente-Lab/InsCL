# arr: task order
# data_dir: test dataset path
arr=('Classification' 'Text_Quality_Evaluation' 'Code' 'Detection' 'Sentiment_Analysis' 'Comprehension' 'Closed_QA' 'Extraction' 'Dialogue' 'Program_Execution' 'Rewriting' 'Open_QA' 'Misc' 'Generation' 'Summarization' 'Mathematics')
data_dir='dataset/test'
model_dir='curriculum_model'
output_merge_dir="curriculum_model_output"

for i in {0..15}
do
echo "start evaluate ${arr[i]}_model"
if [ $i != 0 ]; then
max=$(($i-1))
for d in $(seq 0 ${max})
do
echo "start evaluate ${arr[i]}_model with ${arr[d]}"
torchrun --nproc_per_node 8 --nnodes 1  evaluation_CL.py \
    --model_name_or_path ${model_dir}/${arr[i]} \
    --input_file ${data_dir}/${arr[d]}.jsonl \
    --output_merge_dir ${output_merge_dir} \
    --output_dir ${arr[i]}_model \
    --output_file ${arr[d]} \
    --batch_size 8 \
    --model_max_length 2048
done
fi
echo "start evaluate ${arr[i]}_model with ${arr[i]}"
torchrun --nproc_per_node 8 --nnodes 1  evaluation_CL.py \
    --model_name_or_path ${model_dir}/${arr[i]} \
    --input_file ${data_dir}/${arr[i]}.jsonl \
    --output_merge_dir ${output_merge_dir} \
    --output_dir ${arr[i]}_model \
    --output_file ${arr[i]} \
    --batch_size 8 \
    --model_max_length 2048
done