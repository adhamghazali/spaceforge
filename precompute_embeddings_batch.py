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
    print("calculating text embedding for batch")
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

    batch_size=256
    counter=0
    batch_keys=[0]*batch_size
    batch_caps=['']*batch_size
    batch_images=[[]]*batch_size
    number_of_batches=int(count/batch_size)

    samples=[[0],[''],[[]],[[]]]*(number_of_batches*batch_size)

    for key,img, cap in tqdm(src, total=count, desc=f"Extracting {inp}"):
        batch_keys[counter%batch_size]=key
        batch_caps[counter%batch_size]=cap
        batch_images[counter%batch_size]=img

        if counter%batch_size==0:
            embeddings=get_emb_tensor_batch(batch_caps)
            for ii,_ in enumerate(batch_keys):
                
                embed=get_emb_tensor(embeddings, ii)
                samples[counter]=[batch_keys[ii],batch_images[ii],batch_caps[ii],embed]
        
            #batch_images=[]
            #batch_caps=[]
            #batch_keys=[]
        counter+=1

        
    random.shuffle(samples)
    if os.path.exists(output):
        os.remove(output)

    with wds.TarWriter(output, encoder=True) as dst:
        for sample in tqdm(samples, total=count, desc=f"Writing {output}"):
            dst.write({"__key__":sample[0], "png":sample[1], "txt":sample[2], "npy":sample[3]})






input_shards = braceexpand.braceexpand("/datadrive4T/cc2m/cc12m/{01241..01241}.tar")
output_shards = braceexpand.braceexpand("../{01241..01241}.tar")


for input_shard, output_shard in zip(input_shards, output_shards):
    shuffle_augment_wds(input_shard, output_shard)


