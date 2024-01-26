# Relightable Detailed Human Reconstruction from Sparse Flashlight Images

Note: this repo is still under construction.

## Usage

### Create environment

```shell
git clone https://github.com/Jarvisss/Relightable_human_recon.git 
cd Relightable_human_recon
conda create -n relit_hmr python=3.9 && conda activate relit_hmr
conda install -c conda-forge igl
## install pytorch 2.0.1 with cu118
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
## install kaolin
pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html
## install other dependencies
pip install -r requirements.txt
```

### Training and testing
For pretraining on a human scan dataset:
```shell
bash pretrain.sh 
```

For joint optimization on custom data:
```shell
bash joint_optimize.sh
```

The full parameters are listed in `options/options.py`.
A graphics card with at least 24 GB of memory is recommended. (E.g. RTX 3090)

### Data
Please refer to the `data` folder for for information on the data structure.

### Camera convention
We use the OpenGL camera convention for the rendering of synthetic data for the pretraining phase. Conversely, the *OpenCV camera convention* is adopted for the joint optimization phase.

## Acknowledgements

We would like to thank the authors of the following works to open-source their wonderful projects.

- [IDR](https://github.com/lioryariv/idr)
- [NeuS](https://github.com/Totoro97/NeuS)
- [PIFu](https://github.com/shunsukesaito/PIFu)
- [IRON](https://github.com/Kai-46/IRON)

Thanks for the open-sourcing of dataset [Thuman 2.0](https://github.com/ytrock/THuman2.0-Dataset)