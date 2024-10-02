# Veza između rada "*Denoising Diffusion Probabilistic Models*" i [*PyTorch* implementacije](https://github.com/tcapelle/Diffusion-Models-pytorch)

U ovom izveštaju će biti opisane metode principi modela probabilističke difuzije za uklanjanje šuma (eng. *DDPM* - *Denoising Diffusion Probabilistic Models*) i biće povezane najznačajnije formule sa njihovom implementacijom. Repozitorijum [*Diffusion-Models-pytorch*](https://github.com/tcapelle/Diffusion-Models-pytorch) sadrži implementaciju koja prati model opisan u radu "*Denoising Diffusion Probabilistic Models*", i pruža jednostavnu i lako razumljivu implementaciju *DDPM*-a. Ključne metode *DDPM*-a i implementacije su:
 - proces probabilističke difuzije (zašumljavanje)
 - reverzan proces probabilističke difuzije (uklanjanje šuma)
 - funkcija greške (*MSE* - srednja kvadratna greška)
 - uslovljavanje modela (*CFG* - uslovljavanje bez klasifikatora)

## Proces probabilističke difuzije (eng. *forward diffusion process*)

Proces probabilističke difuzije je u radu opisan kao proces zašumljenja koji postepeno dodaje Gausov šum na ulaz u nizu vremenskih koraka sa oznakom $t$. U radu je proces probabilističke difuzije definisan na sledeći način:

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha_t}} \mathbf{x}_{t-1}, (1-\bar{\alpha_t}) \mathbf{I})$$

Ova formula je u kodu implementirana u funkciji `noise_images` u klasi `Diffusion` ([`ddpm.py`](https://github.com/tcapelle/Diffusion-Models-pytorch/blob/e9bab9a1ae1f1745493031f4163427fe884e12fb/ddpm.py#L30C1-L34C68)):
```python
def noise_images(self, x, t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
    Ɛ = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

...

t = diffusion.sample_timesteps(images.shape[0]).to(device)
x_t, noise = diffusion.noise_images(images, t)
```

## Reverzan proces probabilističke difuzije (eng. *reverse diffusion process*, *denoising*)

Cilj DDPM-a je da nauči reverzan proces probabilističke difuzije, tj. da nauči da dobro ukloni šum iz podataka korak po korak. Reverzan proces probabilističke difuzije je u radu definisan na sledeći način:

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma^2_t \mathbf{I})$$

U kodu, proces uklanjanja šuma predstavlja model *UNet* arhitekture koji treba da treniramo. UNet modeluje srednju vrednost $\mu_\theta$, koja se dobije od zašumljene slike i vremenskog koraka $t$. Kod za UNet je opisan u klasi [*UNet*](https://github.com/tcapelle/Diffusion-Models-pytorch/blob/e9bab9a1ae1f1745493031f4163427fe884e12fb/modules.py#L131C1-L197) u *modules.py*. Proces uklanjanja šuma (generisanje slike) je implementiran u funkciji *sample* u *ddpm.py* na sledeći način:
```python
def sample(self, model, n):
    logging.info(f"Sampling {n} new images....")
    model.eval()
    with torch.no_grad():
        # x0 -> početak uklanjanja šuma, slika je zapravo uzorak iz gausove (normalne) raspodele
        # oblik tenzora je (veličina batch-a, 3 kanala slike, visina, širina), u ovom slučaju su visina i širina iste
        x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            t = (torch.ones(n) * i).long().to(self.device)
            # Prethodna vrednost se propušta kroz UNet koji uklanja šum
            predicted_noise = model(x, t) 
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    model.train()
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x
```

### Funkcija greške

Funkcija greške se koristi kako bi se minimizovala razlika imeđu pravog šuma i šuma koji predvidi model:

$$
L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2 \right]
$$

[U kodu](https://github.com/tcapelle/Diffusion-Models-pytorch/blob/e9bab9a1ae1f1745493031f4163427fe884e12fb/ddpm.py#L67) se za funkciju greške (srednja kvadratna greška), koristi funkcija koja je dostupna iz `torch.nn` modula:
```python
mse = nn.MSELoss()
```

### *DDPM* sa uslovljavanjem

Repozitorijum u kom je implementiran *DDPM* takođe sadrži implementaciju modela sa uslovljavanjem uz pomoć medote uslovljavanja bez klasifikatora (eng. *CFG* - *Classifier Free Guidance*). Uz pomoć CFG, informacija o labeli klase se može koristiti u procesu uklanjanja šuma. Informacija o klasi se koristi u rezidualnim konekcijama i u uskom grlu *UNet_conditional* modela, kako bi imali bolju kontrolu nad time koja se klasa iz skupa podataka generiše na kraju procesa uklanjanja šuma.

### Rezime

Implementacija iz repozitorijuma [*Diffusion-Models-pytorch*](https://github.com/tcapelle/Diffusion-Models-pytorch) blisko prati rad "*Denoising Diffusion Probabilistic Models*". Pruža jednostavnu i jasnu implementaciju za ključne algoritme i koncepte *DDPM*-a:
- [proces probabilističke difuzije](https://github.com/tcapelle/Diffusion-Models-pytorch/blob/e9bab9a1ae1f1745493031f4163427fe884e12fb/ddpm.py#L30C1-L34C68)
- [reverzan proces probabilističke difuzije](https://github.com/tcapelle/Diffusion-Models-pytorch/blob/e9bab9a1ae1f1745493031f4163427fe884e12fb/modules.py#L131C1-L197)
- [funkcija greške](https://github.com/tcapelle/Diffusion-Models-pytorch/blob/e9bab9a1ae1f1745493031f4163427fe884e12fb/ddpm.py#L67)
- [uslovljavanje modela](https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/ddpm_conditional.py)