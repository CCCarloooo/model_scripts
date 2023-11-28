run_llava() {
    local model_path="$1"
    local save_directory="$2"
    local cuda_device="$3"
    echo "Running llava.sh with model_path: $model_path, save_directory: $save_directory, cuda_device: $cuda_device"
    bash llava.sh --model_path "$model_path" --save_directory "$save_directory" --cuda_device "$cuda_device"
    echo "Completed llava.sh with model_path: $model_path, save_directory: $save_directory, cuda_device: $cuda_device"
}

# 定义模型路径、保存目录和CUDA设备
model_path1="/mnt/data2/mxdi/archive/hf-mirror/llava-v1.5-7b"
save_directory1="/mnt/data2/mxdi/models/llava-7b-origin-vicuna"
cuda_device1=0

model_path2="/mnt/data2/mxdi/archive/hf-mirror/llava-v1.5-7b"
save_directory2="/mnt/data2/mxdi/models/llava-7b-origin-vicuna"
cuda_device2=1

# 并行运行任务函数
run_llava "$model_path1" "$save_directory1" "$cuda_device1" &  # 后台运行任务一
run_llava "$model_path2" "$save_directory2" "$cuda_device2" &  # 后台运行任务二

# 等待所有任务完成
wait
echo "All tasks completed"
