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


#dask.config.set(pool=ThreadPool(4))

command='sudo chmod 777 /datadrive4T/cc12m_w_embeds/'
os.system(command)
def get_emb_tensor_batch(texts):
    text_embeds = t5_encode_text([texts], name="google/t5-v1_1-xl", return_attn_mask=False)
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
        wds.to_tuple("__key__", "jpg;png", "txt"),
        wds.map_tuple(None, None, None)
    )

    texts=[]
    for key, _ ,cap in src:
        texts.append(cap)
    print(texts)
    

  



input_shards = braceexpand.braceexpand("/datadrive4T/cc2m/cc12m/{01242..01242}.tar")
output_shards = braceexpand.braceexpand("/datadrive4T/cc12m_w_embeds/{01242..01242}.tar")




for input_shard, output_shard in zip(input_shards, output_shards):
    shuffle_augment_wds(input_shard, output_shard)


