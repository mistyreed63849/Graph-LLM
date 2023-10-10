# GraphLLM: Boosting Graph Reasoning Ability of Large Language Model

This is the implementation for the paper [GraphLLM: Boosting Graph Reasoning Ability of Large Language Model](https://arxiv.org/abs/2310.05845).

## Setup
- You may need a single 80G GPU to run the experiment. We experiment on CUDA 11.8 and torch 2.0.1.
- Setup up a new conda env and install necessary packages.
```
conda create -n graph_llm python=3.10 -y
pip install -r requirements.txt
```
- To run the code, you need the checkpoint and tokenizer of LLaMA-2-7B, which you can access at [Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
After downloading LLaMA-2-7B, soft link the checkpoint folder and the tokenizer folder to the folder of this repository:
```markdown
ln -s /folder/of/LLaMA-2-7B/checkpoint ./LLaMA-7B-2
ln -s /folder/of/LLaMA-2-7B/tokenizer ./Llama-2-7b-hf
```
- Remember to replace the directory `/folder/of/LLaMA-2-7B/checkpoint` and `/folder/of/LLaMA-2-7B/tokenizer` with actual directories!
- The four graph reasoning datasets are [available](https://drive.google.com/file/d/1fRXdCMHpkb1-kuzcxgZPKkILEWBSbW4M).
You may download it and place the zip file in the directory of this repository. And then run the command:
```markdown
unzip dataset.zip -d ./dataset
```
## Get Start

Train and evaluate the model with default settings on graph reasoning datasets on GPU 0:

1. Substructure Counting
```markdown
./scripts/sc.sh
```
2. Maximum Triplet Sum
```markdown
./scripts/mts.sh
```
3. Shortest Path
```markdown
./scripts/sp.sh
```
4. Bipartite Graph Matching
```markdown
./scripts/bgm.sh
```

More hyperparameter settings are at `config.py`

Hyperparameter explanation:
- `--n_encoder_layers` number of transformer layers of textual encoder
- `--n_decoder_layers` number of transformer layers of textual decoder
- `--n_mp_layers` number of graph transformer layers
- `--adapter_dim` hidden dimention
- `--adapter_len` number of prefix tokens per LLM layer
- `--rrwp` positional encoding dimention
- `--batch_size` batch size in memory during training
- `--grad_steps` grad_step $\times$ batch_size = batch size for optimization
- `--lr` the learning rate
- `--num_epochs` number of training epochs
- `--warmup_epochs` number of linear warmup epochs
- `--wd` weight decay