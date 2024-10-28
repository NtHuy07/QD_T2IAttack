# Diverse Adversarial Samples for Text-to-Image Generation via Quality-Diversity Optimization
[![MIT licensed](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

Thai Huy Nguyen, Ngoc Hoang Luong

## Install and setup

### CLIP
Please follow the guidelines in [CLIP Github Repository](https://github.com/openai/CLIP) to install CLIP


### DALL•E Mini
Run the following command to install DALL•E Mini:
```
pip install min-dalle
```
Create and go into the following folder:
```
mkdir target_model/pretrained
cd target_model/pretrained
```
Download and uncompress the files: [dalle-bart](https://drive.google.com/file/d/1Qq_FARjdZlHra3r_g2ZvMLyDPzsgx6Af/view?usp=sharing) and [vqgan](https://drive.google.com/file/d/1ckxflXZnnWJzRvFHhpzuj11Pxr07Wby0/view?usp=sharing)


### Imagen
Download and install the Imagen model:
```
mkdir target_model/pretrained
cd target_model/pretrained
git-lfs clone https://huggingface.co/Cene655/ImagenT5-3B
pip install git+https://github.com/cene555/Imagen-pytorch.git
```

Download and install Real-ESRGAN:
```
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr facexlib gfpgan
pip install -r requirements.txt
python setup.py develop
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
cd ../../..
```

### Word2Vec
Create and go into the folder:
```
mkdir Word2Vec 
cd Word2Vec
```
Download the files: [word2id.pkl](https://drive.google.com/file/d/11kSfFGm1YOo5N08GGytnZy4cMpDTyd0h/view?usp=sharing) and [wordvec.pkl](https://drive.google.com/file/d/1h1hhkyZWZc-JhKqJBPtnJ2riooXMY-e0/view?usp=sharing)

### Other dependencies
Install dependencies:
```
pip install -r requirements.txt
```

## Run experiments
To run the attack with MAP-Elites:
```
python run_me_attack.py --ori_sent [original sentence] --tar_sent [target sentence] --tar_img_path [target image path] --tar_model [target model] --log_path [root log path] --log_save_path [log save path] --intem_img_path [intermediate results save path] --save_all_images [save every generated image during search]
```

An example of running the attack with MAP- on DALLE-Mini model:
```
python run_me_attack.py --ori_sent "A couple of people in the snow with skis." --tar_sent "A beautiful cake sets on a table that says, \"Happy Birthday\"." --tar_img_path "./target.png" --tar_model "dalle-mini" --save_all_images
```

To run the attack with normal GA:
```
python run_ga_attack.py --ori_sent [original sentence] --tar_sent [target sentence] --tar_img_path [target image path] --tar_model [target model] --log_path [root log path] --log_save_path [log save path] --intem_img_path [intermediate results save path] --best_img_path [output best images save path] --save_all_images [save every generated image during search]
```


## Acknowledgements

Our source code is built upon:
- [RIATIG: Reliable and Imperceptible Adversarial Text-to-Image Generation with Natural Prompts](https://github.com/WUSTL-CSPL/RIATIG) by Han Liu, Yuhao Wu, Shixuan Zhai, Bo Yuan, Ning Zhang
- [OpenELM](https://github.com/CarperAI/OpenELM.git)
