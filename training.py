import torch
import torch.optim as optim
import os
from model import build_model
from args import get_args
import csv
from utils import show_batch

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model එක සහ Optimizer එක මෙතැනදී සූදානම් කරගන්නවා
model = build_model(args.backbone)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def validate_model(model, val_loader, device):
    model.train() # Faster R-CNN වල loss එක ගන්න නම් train mode එකේම තියෙන්න ඕනේ
    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            # 1. පින්තූර ටික device එකට යැවීම
            images = list(img.to(device) for img in images)
            
            # 2. Targets ටික නිවැරදිව device එකට යැවීම
            # මෙතැනදී 't' කියන්නේ dictionary එකක්.
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 3. Model එකෙන් loss එක ලබා ගැනීම
            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    # 0 වලින් බෙදීම වළක්වා ගැනීමට
    return val_loss_sum / val_count if val_count > 0 else 0

def train_model(model, train_loader, val_loader, device):
    global best_val_loss
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            # පින්තූර ටික device එකට යැවීම
            images = list(image.to(device) for image in images)
            
            # Target එකේ තියෙන boxes සහ labels නිවැරදිව device එකට යැවීම
            # මෙතැනදී 't' කියන්නේ ලැයිස්තුවේ තියෙන එක dictionary එකක්
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            #show_batch(images, targets)
            optimizer.zero_grad()

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = validate_model(model, val_loader, device)

        # හොඳම model එක save කිරීම
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")
        log_file = os.path.join(args.out_dir, "training_log.csv")
        file_exists = os.path.isfile(log_file)

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
            writer.writerow([epoch + 1, train_epoch_loss, val_loss])