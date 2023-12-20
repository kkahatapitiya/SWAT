# SWAT
Code for our IJCAI 2023 paper "SWAT: Spatial Structure Within and Among Tokens"


This project is based on the original repo at: https://github.com/rwightman/pytorch-image-models


Changes to code are marked within:
----------------------------------
########## START ###########
########### END ############


Structure-aware tokenization implemented in:
--------------------------------------------
./timm/models/layers/patch_embed.py


MLP --> Conv implemented in:
----------------------------
./timm/models/layers/mlp.py


Structure-aware Mixing implemented in:
--------------------------------------
./timm/models/mlp_mixer.py
./timm/models/vision_transformer.py
./timm/models/swin_transformer.py


FLOP counting in:
-----------------
./count_flops.py


config files in:
----------------
./config/mixer_i1k_scratch.yaml
./config/vit_i1k_scratch.yaml
./config/swin_i1k_scratch.yaml


To run the code on 8 gpus:
--------------------------
download the imagenet-1k dataset, and run,
./distributed_train.sh 8 <IMAGENET_1K_DIR> --config <CONFIG_FILE>
