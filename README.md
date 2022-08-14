# spaceforge

A training pipeline for Imagen,  Text-to-Image Neural Network

Training runs are logged in Wandb: https://wandb.ai/adham



#### Plan: 

- [x] Do initial tests
- [ ] Train the folowing net 
  - [ ] sf-v1-1 : `256x256` on [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) 250K iterations
  - [ ] sf-v1-1 `512x512` on [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution)  (200K iterations) SR
- [ ] Resume training from  sf-v1-1 to sf-v1-2
  - [ ] sf-v1-2 : resumed from on laion-improved-aesthetics (a subset of laion 2B  with `512x512` and  improved aesthetics  score>5.0 and estimated watermark prob< 0.5)
    - [ ] for score use [improved aesthetics estimator](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [ ] Resume training from sf-v1-2 to sf-v1-3
  - [ ] 10% dropping of the text conditioning 



#### Todo: 

- [ ] Make sure training is running smoothly
- [ ] Download the following datasets:
  - [ ]  [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) 
  - [ ]  [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution)  
- [ ] Figure out which vms we are going to use
- [ ]  use [improved aesthetics estimator](https://github.com/christophschuhmann/improved-aesthetic-predictor) to add meta data
- [ ] Tests



#### Datasets

##### Conceptual 12M

Downloaded with [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md)

On Linux ...
```bash
wget https://storage.googleapis.com/conceptual_12m/cc12m.tsv
```
```bash
sed -i "1s/^/url\tcaption\n/" cc12m.tsv
```
```bash
img2dataset --url_list cc12m.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder cc12m --processes_count 8 --thread_count 32 --image_size 256\
             --enable_wandb True
```





#### Download 2B Images dataset - Laion2B-en Joined with watermark and punsafe data

```markdown
mkdir laion2B-en && cd laion2B-en
for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en-joined/resolve/main/part-$i-4cfd6e30-f032-46ee-9105-8696034a8373-c000.snappy.parquet; done
cd ..
```





Guidelines: 

- Larger language models seem to be better

- Unet size has little impact on quality

- Training data quality and quantity seems to have the largest impact

  



#### Some running notes

- Batch size of 64-512 seems to be good.
- Setting `max_grad_norm = 1.25` makes training more stable, but appears to considerably slow convergence and hurt performance.
- Best results have been attained with a learning rate of around 1.5e-5 when combined with batch size of 256.
