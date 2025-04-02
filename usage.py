import pickle as pkl
import torch
from model import IeGenerator
from train import train

MODELS = {
    "scibert": "scibert_cased",
    "scibert_cased": "scibert_cased",
    "scibert_uncased": "scibert_uncased",
    "scibert_uncased_cased": "scibert_uncased_cased",
    "scibert_uncased_cased_large": "scibert_uncased_cased_large",
    "deberta-l": "microsoft/deberta-v3-large",
    "deberta": "microsoft/deberta-v3-base",
    "deberta-s": "microsoft/deberta-v3-small",
    "deberta-xs": "microsoft/deberta-v3-xsmall"
}

with open("dataset/conll04.pkl", "rb") as f:
    data = pkl.load(f)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Load mappings
class_to_id = data['span_to_id']  # entity to id mapping
rel_to_id = data['rel_to_id']  # relation to id mapping
rel_to_id["stop_entity"] = len(rel_to_id)  # add a new relation for stop entity

model = IeGenerator(
    class_to_id, rel_to_id, model_name=MODELS["deberta-xs"], max_width=20,
    num_prompts=5, span_mode="conv_share", use_pos_code=True, p_drop=0.08, cross_attn=True
)

model.to(device)

optimizer = torch.optim.Adam([
    # encoder
    {'params': model.token_rep.parameters(), 'lr': 3e-5},

    # decoder
    {'params': model.decoder.parameters(), 'lr': 7e-5},

    # lstm
    {'params': model.rnn.parameters(), 'lr': 1e-4},

    # projection layers
    {'params': model.project_memory.parameters(), 'lr': 1e-4},
    {'params': model.project_queries.parameters(), 'lr': 1e-4},
    {'params': model.project_tokens.parameters(), 'lr': 1e-4},
    {'params': model.span_rep.parameters(), 'lr': 1e-4},
    {'params': model.project_span_class.parameters(), 'lr': 1e-4},
    {'params': model.embed_proj.parameters(), 'lr': 1e-4},
])

train(
    model=model, optimizer=optimizer, train_data=data['train'], eval_data=data['dev'],
    train_batch_size=32, eval_batch_size=32,
    n_epochs=15, n_steps=20000, warmup_ratio=0.15,
    grad_accumulation_steps=1,
    max_num_samples=10,
    save_interval=1000, log_dir="model_ckpts"
)

