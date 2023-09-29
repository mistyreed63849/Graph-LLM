import pickle
from config import module_path
import torch
from transformers import LlamaTokenizer
from torch.utils.data import TensorDataset
import datasets



def load_textual_sp():
    dataset = datasets.load_from_disk(f"{module_path}/dataset/sp")
    split = pickle.load(open(f"{module_path}/dataset/sp/sp_split.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/sp/sp_edge.pkl", 'rb'))
    return dataset, split, edge_index


def load_textual_mts():
    dataset = datasets.load_from_disk(f"{module_path}/dataset/mts")
    split = pickle.load(open(f"{module_path}/dataset/mts/mts_split.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/mts/mts_edge.pkl", 'rb'))
    return dataset, split, edge_index


def load_textual_sc():
    dataset = datasets.load_from_disk(f"{module_path}/dataset/sc")
    split = pickle.load(open(f"{module_path}/dataset/sc/sc_split.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/sc/sc_edge.pkl", 'rb'))
    return dataset, split, edge_index


def load_textual_bgm():
    dataset = datasets.load_from_disk(f"{module_path}/dataset/bgm")
    split = pickle.load(open(f"{module_path}/dataset/bgm/bgm_split.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/bgm/bgm_edge.pkl", 'rb'))
    return dataset, split, edge_index




