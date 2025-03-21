# Consistency models repository

## 2D dataset study

- The code is available in the notebook `./2D_dataset/project-2Ddata.ipynb` as well as the training cells (with outputs).
- All figures are availables in the `./figures` folder.
    - Figures regarding the output of the degradation process are in `./figures/degradation_model` folder.
    - Figures created to probe training by sampling are in different folders (one folder per training, model and hyperparameters in the folder name).
    - Gif of the different training are availble.
    - Figures from the report for the 2D part are gathered in this folder too.
- All trained models are available in `./models` folder.

## Using our code - Sampling and inpainting

We provide examples of multistep generation and image inpainting in scripts/launch.sh. Here are the two command lines we used to generate images (sampling, first command line) or do inpainting (second command line):

- `consistency_models/scripts$ python image_sample.py --batch_size 8 --training_mode consistency_distillation --sampler multistep --ts 0,17,39 --steps 40 --model_path /home/onyxia/work/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples 5 --resblock_updown True --use_fp16 False --weight_schedule uniform`

- `consistency_models/scripts$ python image_sample_inpainting.py --batch_size 7   --training_mode consistency_distillation   --sampler multistep   --ts 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40   --steps 40   --model_path /home/onyxia/work/checkpoints/cd_bedroom256_lpips.pt   --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0   --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --weight_schedule uniform`