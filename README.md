# Image reconstruction on CIFAR-10 using unconditional DDPM and RePaint inpainting method

## Training a model

You can train a model on a custom dataset using `train_ddpm.ipynb`

## Model evaluation

Your model can be evaluated by running `python eval_repaint.py`. You need to have a ground truth directory named `gt` for the ground truths and a directory for masks named `mask` in which the ground truth and mask images will be located.

## Pretrained model

The model trained using this repository, which can be used with HuggingFace diffusers, can be found at [https://huggingface.co/ikkjo/ddpm-cifar10-64](https://huggingface.co/ikkjo/ddpm-cifar10-64)