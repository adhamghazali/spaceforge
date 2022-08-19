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
def get_emb_tensor(text):
    text_embeds = t5_encode_text([text], name="google/t5-v1_1-xl", return_attn_mask=False)
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


input_shards = braceexpand.braceexpand("/datadrive4T/cc2m/cc12m/{01026..01242}.tar")
output_shards = braceexpand.braceexpand("/datadrive4T/cc12m_w_embeds/{01026..01242}.tar")

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
