import torch
from PIL import Image
import torchvision.transforms.functional as F
from PIL import ImageOps

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, size=640):
        self.df = df.reset_index(drop=True)
        self.size = size # අපි පින්තූර පත් කරන ප්‍රමාණය

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. පින්තූරය විවෘත කර Resize කිරීම
        img = Image.open(row["images"]).convert("RGB")
        img = ImageOps.exif_transpose(img)
        original_w, original_h = img.size
        
        # පින්තූරය නව ප්‍රමාණයට (640, 640) වෙනස් කිරීම
        img = img.resize((self.size, self.size))
        image = F.to_tensor(img)

        boxes, labels = [], []
        
        # 2. Labels කියවා Resize එකට අනුව ඛණ්ඩාංක වෙනස් කිරීම
        with open(row["labels"]) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                
                # මුල් පින්තූරයේ pixel අගයන් ගණනය කිරීම
                x1 = (xc - bw/2) * original_w
                y1 = (yc - bh/2) * original_h
                x2 = (xc + bw/2) * original_w
                y2 = (yc + bh/2) * original_h
                
                # නව ප්‍රමාණයට (640x640) අනුපාතය අනුව වෙනස් කිරීම
                x1 = x1 * (self.size / original_w)
                y1 = y1 * (self.size / original_h)
                x2 = x2 * (self.size / original_w)
                y2 = y2 * (self.size / original_h)
                
                # වැදගත්: x2 සහ y2 අගයන් x1 සහ y1 ට වඩා වැඩි බව තහවුරු කර ගැනීම
                if (x2 > x1) and (y2 > y1):
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls) + 1)

        # පෙට්ටි නොමැති නම් (empty boxes) ගැටලුවක් නොවීමට
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }
        else:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }

        return image, target