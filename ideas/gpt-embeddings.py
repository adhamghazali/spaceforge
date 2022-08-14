# do this once to precompute embeddings and save to .npy
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit")
model = AutoModel.from_pretrained("Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.eval()


def generate_embeddings(texts):
    batch_tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    batch_tokens.to(device)

    with torch.no_grad():
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

    weights = torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(0).unsqueeze(-1).expand(last_hidden_state.size()).float().to(last_hidden_state.device)

    input_mask_expanded = batch_tokens["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()

    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask
    return embeddings
    

texts = [
    "drawing of a girl with red hair",
    "drawing of a girl with green hair",
    "drawing of a girl with blue hair",
    "drawing of a catgirl",
    "drawing of a catgirl with green hair",
    "drawing of a black haired girl",
]

embeddings = generate_embeddings(texts)
embeddings_np = embeddings.cpu().detach().numpy()
np.save("/mnt/k/ml/datasets/danbooru2021/experiment/sample_text_embeds.npy", embeddings_np)


# how to sample, make sure your imagen/unets have text_embed_dim set to 2048
def sample_images(trainer, unet_number=1):
    precomputed_embeds = np.load("/mnt/k/ml/datasets/danbooru2021/experiment/sample_text_embeds.npy")
    text_embeds = torch.from_numpy(precomputed_embeds)
    text_embeds = torch.tile(text_embeds, (5, 1))  # repeat texts 5 times
    text_embeds = torch.unsqueeze(text_embeds, 1)

    rng_state = torch.random.get_rng_state()
    torch.manual_seed(42)
    images = trainer.sample(text_embeds=text_embeds, cond_scale=7, stop_at_unet_number=unet_number)
    torch.random.set_rng_state(rng_state)
    return make_grid(images, nrow=5)