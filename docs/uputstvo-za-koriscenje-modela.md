# Uputstvo za korišćenje [modela DDPM](https://github.com/tcapelle/Diffusion-Models-pytorch)


## 1. Preuzimanje repozitorijuma

`git clone https://github.com/tcapelle/Diffusion-Models-pytorch`

## 2. Podešavanje okruženja

1. Napravi novo virtuelno okruženje

    `python -m venv ddpm_venv`

2. Aktiviraj novo okruženje:

    - Windows - `.\ddpm_venv\Scripts\activate.bat` (cmd) ili `.\ddpm_venv\Scripts\activate.ps1` (PowerShell)
    - Linux - `source ddpm_venv/bin/activate`

3. Instaliraj potrebne pakete:

    `pip install kaggle torch wandb torchvision Pillow fastdownload matplotlib fastcore`

__Napomena__:
- Ukoliko želiš pokretati treniranje modela sa uslovljavanjem i primere iz Jupyter svesaka, potrebno je ispratiti [*uputsvo za instalaciju **wandb** paketa*](https://docs.wandb.ai/quickstart/#:~:text=Create%20an%20account%20and%20install,Python%203%20environment%20using%20pip%20.)

## 3. Treniranje difuzionog modela na sopstvenim podacima:

Preuzimanje cifar10 skupa podataka je opisano u `00_prepare_data.ipynb`, a treniranje modela je opisano u `01_train_cifar.ipynb`

### Treniranje modela bez uslovljavanja (`ddpm.py`):
```bash 
# (opciono) Konfigurisati parametre u ddpm.py
# Podesiti putanju do skupa podataka u ddpm.py, zatim pokrenuti
python ddpm.py
```

### Treniranje modela sa uslovljavanjem (`ddpm_conditional.py`):
```bash
# (opciono) Konfigurisati parametre u ddpm_conditional.py
# Podesiti putanju do skupa podataka u ddpm_conditional.py, zatim pokrenuti
python ddpm_conditional.py
```

## 4. Generisanje novih slika

Generisanje novih slika je opisano u `02_generate_samples.ipynb`

U kodu ispod se nalazi sve potrebno za generisanje novih slika ukoliko se uključe klase ddpm.py odnosno ddpm_conditional.py:

### Model bez uslovljavanja

Model bez uslovljavanja je treniran na ["Landscape" skupu podataka](https://www.kaggle.com/datasets/arnaud58/landscape-pictures). 

```python
device = "cuda"
model = UNet().to(device)
ckpt = torch.load("unconditional_ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
x = diffusion.sample(model, n=16)
plot_images(x)
```

### Model sa uslovljavanjem
Model sa uslovljavanjem je treniran na CIFAR-10 64x64 skupu podataka sa klasama: [airplane:0, auto:1, bird:2, cat:3, deer:4, dog:5, frog:6, horse:7, ship:8, truck:9]

```python
n = 4
device = "cuda"
model = UNet_conditional(num_classes=10).to(device)
ckpt = torch.load("conditional_ema_ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=64, device=device)
y = torch.Tensor([6] * n).long().to(device)
x = diffusion.sample(model, n, y, cfg_scale=3)
plot_images(x)
```

## 5. Pretrenirani modeli

Pretrenirane modele (sa uslovljavanjem i bez uslovljavanja) možeš preuzeti [ovde](https://drive.google.com/drive/folders/1beUSI-edO98i6J9pDR67BKGCfkzUL5DX?usp=sharing).