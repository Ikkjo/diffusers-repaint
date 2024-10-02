import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pytorch_lightning as pl
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import csv


class ImagePairsDataset(Dataset):
    def __init__(self, inpainted_dir, ground_truth_dir, transform=None):
        """
        Args:
            inpainted_dir (str): Directory with inpainted images.
            ground_truth_dir (str): Directory with ground truth images.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.inpainted_dir = inpainted_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.inpainted_images = sorted(os.listdir(inpainted_dir))
        self.ground_truth_images = sorted(os.listdir(ground_truth_dir))
        os.makedirs("results", exist_ok=True)
    
    def __len__(self):
        return len(self.inpainted_images)
    
    def __getitem__(self, idx):
        inpainted_image_path = os.path.join(self.inpainted_dir, self.inpainted_images[idx])
        ground_truth_image_path = os.path.join(self.ground_truth_dir, self.ground_truth_images[idx])
        
        inpainted_image = Image.open(inpainted_image_path).convert("RGB")
        ground_truth_image = Image.open(ground_truth_image_path).convert("RGB")
        
        if self.transform:
            inpainted_image = self.transform(inpainted_image)
            ground_truth_image = self.transform(ground_truth_image)
        
        return inpainted_image, ground_truth_image


class LPIPSEvaluator(pl.LightningModule):
    def __init__(self, inpainted_dir, ground_truth_dir, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ])
        self.dataset = ImagePairsDataset(inpainted_dir, ground_truth_dir, transform=self.transform)
        self.lpips_metric = LPIPS(net_type='alex', reduction='mean').to(self.device)
        self.test_step_outputs = []

    def setup(self, stage=None):
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def test_step(self, batch, batch_idx):
        inpainted_images, ground_truth_images = batch
        inpainted_images = inpainted_images.to(self.device)
        ground_truth_images = ground_truth_images.to(self.device)
        lpips_scores = self.lpips_metric(inpainted_images, ground_truth_images)

        self.test_step_outputs.append(lpips_scores)
        
        for score in self.test_step_outputs:
            self.log("lpips", score, prog_bar=True)
        
        return {"lpips": [lpips_scores]}

    def on_test_epoch_end(self):
        # Calculate average LPIPS across all batches
        all_scores = torch.tensor(self.test_step_outputs)
        avg_lpips = all_scores.mean()
        
        self.log("avg_lpips", avg_lpips)
        print(f"Average LPIPS score: {avg_lpips.item()}")

        with open("results/lpips_scores.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Index", "LPIPS Score"])
            for idx, score in enumerate(all_scores):
                writer.writerow([idx, score.item()])
            writer.writerow(["Average", avg_lpips.item()])
        
        return avg_lpips

    def test_dataloader(self):
        return self.dataloader


if __name__ == "__main__":
    # Set up paths to inpainted and ground truth image directories
    inpainted_dir = "inpainted"
    ground_truth_dir = "gt"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluator = LPIPSEvaluator(inpainted_dir=inpainted_dir, ground_truth_dir=ground_truth_dir, batch_size=64)
    trainer = pl.Trainer(accelerator=device, devices=1)
    trainer.test(evaluator)