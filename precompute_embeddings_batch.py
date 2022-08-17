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
    text_embeds = t5_encode_text(texts, name="google/t5-v1_1-xl", return_attn_mask=False)
    #print(text_embeds.cpu().dtype)
    return text_embeds.cpu().detach().numpy()

def get_emb_tensor(embeddings,key):
    return embeddings[key]


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
        wds.to_tuple("__key__", "jpg;png", "txt",),
        wds.map_tuple(None, None, None)
    )

    texts=[]
    batch_size=16
    counter=0
    batch_keys=[]
    CAPS=[]
    IMAGES=[]
    samples=[]

    for key,img, cap in tqdm(src, total=count, desc=f"Extracting {inp}"):
        texts.append(cap)
        counter+=1
        batch_keys.append(key)
        CAPS.append(cap)
        IMAGES.append(img)

        if counter==batch_size:
            embeddings=get_emb_tensor_batch(texts)
            for ii,_ in enumerate(batch_keys):
                
                embed=get_emb_tensor(embeddings, ii)
                samples.append([batch_keys[ii],IMAGES[ii],CAPS[ii],embed])
            texts=[]
            IMAGES=[]
            CAPS=[]
            batch_keys=[]





    
    

  



input_shards = braceexpand.braceexpand("/datadrive4T/cc2m/cc12m/{01241..01241}.tar")
output_shards = braceexpand.braceexpand("/datadrive4T/cc12m_w_embeds/{01241..01241}.tar")




for input_shard, output_shard in zip(input_shards, output_shards):
    shuffle_augment_wds(input_shard, output_shard)

