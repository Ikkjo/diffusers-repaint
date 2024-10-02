import torch
from PIL import Image
from diffusers import RePaintPipeline, RePaintScheduler
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class InpaintingDataset(Dataset):
    def __init__(self, input_dir, mask_dir):
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.input_dir, img_name)
        mask_path = os.path.join(self.mask_dir, f"{self.mask_files[idx % 100]}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask, img_name
    

def inpaint_batch(pipeline, batch_images, batch_masks, generator=None):
    inpainted_images = pipeline(
        image=batch_images,
        mask_image=batch_masks,
        num_inference_steps=500,
        eta=1.0,
        jump_length=10,
        jump_n_sample=10,
        generator=generator
    ).images
    return inpainted_images


def main(gt_path, mask_path, device):

    scheduler = RePaintScheduler.from_pretrained("ddpm-cifar10-64/scheduler")
    pipeline = RePaintPipeline.from_pretrained(
        "ddpm-cifar10-64",
        scheduler=scheduler).to(device)
    generator = torch.Generator(device=device).manual_seed(0)

    input_dir = gt_path
    mask_dir = mask_path
    masked_input_dir = "gt_masked"
    output_dir = "inpainted"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masked_input_dir, exist_ok=True)

    dataset = InpaintingDataset(input_dir, mask_dir)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=15)

    for batch_images, batch_masks, batch_names in dataloader:
        batch_images = batch_images.to(device)
        batch_masks = batch_masks.to(device)

        pil_images = [transforms.ToPILImage()(img) for img in batch_images]
        pil_masks = [transforms.ToPILImage()(mask) for mask in batch_masks]
        
        inpainted_images = inpaint_batch(pipeline, pil_images, pil_masks, generator=generator)

        for original, mask, inpainted, name in zip(pil_images, pil_masks, inpainted_images, batch_names):
            gt_masked = Image.composite(original, Image.new('RGB', original.size, color='black'), mask.convert("L"))
            gt_masked.save(os.path.join(masked_input_dir, f"{name}"))
            inpainted.save(os.path.join(output_dir, f"{name}"))

    print("Batch inpainting completed. Results saved in", output_dir)

if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt_path = "gt"
    mask_path = "mask"

    main(gt_path=gt_path, mask_path=mask_path, device=device)
