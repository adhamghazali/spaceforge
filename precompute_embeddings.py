import json
import torch
from PIL import Image
import random
import braceexpand
from time import time
from tqdm import tqdm
import webdataset as wds
from imagen_pytorch.t5 import t5_encode_text
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)



def get_emb_tensor(text):
    text_embeds = t5_encode_text([text], name="google/t5-v1_1-xl", return_attn_mask=False)
    return text_embeds.cpu()


def get_count(input_file):
    stats_file = input_file[:-4] + "_stats.json"
    f = open(stats_file)
    stats = json.load(f)
    f.close()
    count = stats["successes"]
    return count


def shuffle_augment_wds(input, output):
    start = time()
    count = get_count(input)
    input = input
    src = wds.DataPipeline(
        wds.SimpleShardList(input),
        wds.tarfile_to_samples(),
        wds.decode("rgb"),
        wds.to_tuple("__key__", "jpg;png", "txt", "txt"),
        wds.map_tuple(None, None, None, get_emb_tensor)
    )

    samples = []
    for key, img, cap, emb in tqdm(src, total=count, desc=f"Extracting {input}"):
        #print(img)
        
        #Image.open(img)
        samples.append([key, img, cap, emb])
    random.shuffle(samples)

    dst = wds.TarWriter(output, encoder=True)
    for sample in tqdm(samples, total=count, desc=f"Writing {output}"):
        dst.write({"__key__":sample[0], "png":sample[1], "txt":sample[2], "emb.pyd":sample[3]})
    end = time()
    print(f"Finished - {end-start:.0f}s")

#uncomment the following two lines to restore func
#input_shards = braceexpand.braceexpand("/datadrive/cc2m/cc12m/{00005..00005}.tar")
#output_shards = braceexpand.braceexpand("/datadrive4T/cc12m_w_embeds/{00005..00005}.tar")

input_shards=['01242.tar','01242.tar']#['00939.tar','00500.tar','00558.tar','00314.tar','00364.tar','00415.tar','00658.tar','00583.tar','00634.tar','00685.tar','00832.tar','00934.tar','01000.tar','01133.tar','01082.tar', '01184.tar','01242.tar','01242.tar']  #comment this to restore orig func
output_shards=input_shards  #comment this to restore orig func

for input_shard, output_shard in zip(input_shards, output_shards):
    input_shard='/datadrive/cc2m/cc12m/'+input_shard   #comment this to restore orig func
    output_shard='/datadrive/cc2m/cc12m_w_embeds/'+output_shard  #comment this to restore orig func
    shuffle_augment_wds(input=input_shard, output=output_shard)
    
