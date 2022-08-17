import time as T
import json
import tarfile
import torch
from PIL import Image
import random
import braceexpand
from time import time
from tqdm import tqdm
import webdataset as wds
from imagen_pytorch.t5 import t5_encode_text
device =torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#import dask

print(device)
import gopen
import os




import torch
import transformers
from typing import List
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from einops import rearrange

transformers.logging.set_verbosity_error()

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# config

MAX_LENGTH = 256

DEFAULT_T5_NAME = 'google/t5-v1_1-base'

T5_CONFIGS = {}

# singleton globals

def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name)
    return tokenizer

def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model

def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

# encoding text

def t5_tokenize(
    texts: List[str],
    tokenizer,
    name = DEFAULT_T5_NAME
    
):
  

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    return input_ids, attn_mask

def t5_encode_tokenized_text(
    token_ids,
    t5,
    attn_mask = None,
    pad_id = None,
    name = DEFAULT_T5_NAME
    
):
    assert exists(attn_mask) or exists(pad_id)
    #t5, _ = get_model_and_tokenizer(name)

    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())
    #t5=t5model

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = token_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()

    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.) # just force all embeddings that is padding to be equal to 0.
    return encoded_text

def t5_encode_text(
    texts: List[str],
    tokenizer,
    t5model,
    name = DEFAULT_T5_NAME,
    return_attn_mask = False
    
):
    token_ids, attn_mask = t5_tokenize(texts,tokenizer, name = name)
    encoded_text = t5_encode_tokenized_text(token_ids, t5model, attn_mask = attn_mask, name = name)

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask

    return encoded_text



name="google/t5-v1_1-xl"
global t5
global tokenizer
t5, tokenizer = get_model_and_tokenizer(name)

if torch.cuda.is_available():
    t5 = t5.cuda()
device = next(t5.parameters()).device




#dask.config.set(pool=ThreadPool(4))

command='sudo chmod 777 /datadrive4T/cc12m_w_embeds/'
os.system(command)
def get_emb_tensor(text):
    text_embeds = t5_encode_text([text],tokenizer, t5,  name="google/t5-v1_1-xl", return_attn_mask=False)
    #print(text_embeds.cpu().dtype)
    return text_embeds.cpu().detach().numpy()


def get_count(input_file):
    stats_file = input_file[:-4] + "_stats.json"
    f = open(stats_file)
    stats = json.load(f)
    f.close()
    count = stats["successes"]
    return count


def shuffle_augment_wds(inp, output):
    start = time()
    count =get_count(inp)
    #inp = input
    src = wds.DataPipeline(
        wds.SimpleShardList(inp),
        wds.tarfile_to_samples(),
        wds.decode("rgb"),
        wds.to_tuple("__key__", "jpg;png", "txt", "txt"),
        wds.map_tuple(None, None, None, get_emb_tensor)
    )

    samples = []
    for key, img, cap, emb in tqdm(src, total=count, desc=f"Extracting {inp}"):
        samples.append([key, img, cap, emb])
    random.shuffle(samples)
    if os.path.exists(output):
        os.remove(output)

    #fileobj = gopen.gopen(output, "wb")

    #with gopen.gopen(output,"wb") as fileobj:
    with wds.TarWriter(output, encoder=True) as dst:
        for sample in tqdm(samples, total=count, desc=f"Writing {output}"):
            dst.write({"__key__":sample[0], "png":sample[1], "txt":sample[2], "npy":sample[3]})


    end = time()
    print(f"Finished - {end-start:.0f}s")

#uncomment the following two lines to restore func


input_shards = braceexpand.braceexpand("/datadrive4T/cc2m/cc12m/{00000..01242}.tar")
output_shards = braceexpand.braceexpand("/datadrive4T/cc12m_w_embeds/{00000..01242}.tar")

#input_shards= dask.delayed(input_shards)

#results = [dask.delayed(shuffle_augment_wds)(i,j) for i,j in zip(input_shards, output_shards)]
#print(results)
#from concurrent.futures import ThreadPoolExecutor
#with dask.config.set(pool=ThreadPoolExecutor(2)):
#    dask.compute(*results)(num_workers=2)


for input_shard, output_shard in zip(input_shards, output_shards):
    shuffle_augment_wds(input_shard, output_shard)


#with dask.config.set(pool=ThreadPoolExecutor(2)):
#    dask.compute(*results)
#dask.compute(*results)(num_workers=2)    
