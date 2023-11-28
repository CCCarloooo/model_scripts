import argparse
import os
from transformers import LlamaForCausalLM, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch
from accelerate import init_empty_weights


# 创建一个解析器
parser = argparse.ArgumentParser(description='Convert Llama model.')
parser.add_argument('--model_path', type=str, help='Path to the pretrained Llama model')
parser.add_argument('--save_directory', type=str, help='Directory to save the converted model')
parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device to use')

# 解析参数
args = parser.parse_args()

# 拆分保存目录，获取最后的部分
_, save_directory_last_part = os.path.split(args.save_directory)

if '7b' in save_directory_last_part:
    model = LlamaForCausalLM.from_pretrained(args.model_path).to(f"cuda:{args.cuda_device}") 
else:   
    model = LlamaForCausalLM.from_pretrained(args.model_path) 

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

tokenizer.save_pretrained(args.save_directory)
model.save_pretrained(args.save_directory)






