cd /mnt/data2/mxdi/archive/models

python convert_llava.py \
    --model_path /mnt/data2/mxdi/archive/models/cl_llava/llava-sft-v1.5-13b-pretrained-mlp2x_gelu/checkpoint-2000 \
    --save_directory /mnt/data2/mxdi/archive/models/cl_llava_vicuna/2000_13b \
    --cuda_device 4

#检查是否存在转换后的目录
ls /mnt/data2/mxdi/archive/models/cl_llava_vicuna/2000_13b

