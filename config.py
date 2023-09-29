from os import path
import argparse

module_path = path.dirname(path.abspath(__file__))

from utils.dataset import load_textual_mts, load_textual_sc, load_textual_bgm, load_textual_sp
from utils.dataset_preprocess import *


load_dataset = {
    'sc': load_textual_sc,
    'mts': load_textual_mts,
    'sp': load_textual_sp,
    'bgm': load_textual_bgm,
}


preprocess_original_dataset = {
    'sc': preprocess_function_original_sc,
    'mts': preprocess_function_original_mts,
    'sp': preprocess_function_original_sp,
    'bgm': preprocess_function_original_bgm,
}


preprocess_train_dataset = {
    'sc': preprocess_function_generator_sc,
    'mts': preprocess_function_generator_mts,
    'sp': preprocess_function_generator_sp,
    'bgm': preprocess_function_generator_bgm,
}

preprocess_test_dataset = {
    'sc': preprocess_test_function_generator_sc,
    'mts': preprocess_test_function_generator_mts,
    'sp': preprocess_test_function_generator_sp,
    'bgm': preprocess_test_function_generator_bgm,
}

instruction_len = {
    'sp': 50,
    'sc': 45,
    'bgm': 35,
    'mts': 50,
}

original_len = {
    'sp': 60,
    'sc': 60,
    'bgm': 60,
    'mts': 90,
}

task_level = {
    'sp': 'pair',
    'sc': 'node',
    'bgm': 'graph',
    'mts': 'node',
}


def parse_args_llama():
    parser = argparse.ArgumentParser(description="GraphLLM")

    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--exp_num", type=int, default=1)
    # parser.add_argument("--tokenizer_path", type=str, default='Llama-2-7b-hf')
    parser.add_argument("--model_name", type=str, default='LLaMA-7B-2')

    parser.add_argument("--dataset", type=str, default='mol')
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=0.1)


    # parser.add_argument("--num_hops", type=int, default=3)
    parser.add_argument("--adapter_len", type=int, default=5)
    # parser.add_argument("--adapter_layer", type=int, default=16)
    parser.add_argument("--adapter_dim", type=int, default=768)
    parser.add_argument("--adapter_n_heads", type=int, default=6)


    parser.add_argument("--n_decoder_layers", type=int, default=4)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_mp_layers", type=int, default=4)


    # Model Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_steps", type=int, default=2)


    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=15)


    parser.add_argument("--warmup_epochs", type=float, default=1)
    # parser.add_argument("--min_lr", type=float, default=5e-6)

    # ParaEff Tuning
    # parser.add_argument("--w_adapter", default=False, action='store_true')
    # parser.add_argument("--w_lora", default=False, action='store_true')
    # parser.add_argument("--lora_alpha", type=int, default=16)
    # parser.add_argument("--lora_r", type=int, default=8)

    # RRWP
    parser.add_argument("--rrwp", type=int, default=8)

    # Inference
    # parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=32)

    args = parser.parse_args()
    return args



