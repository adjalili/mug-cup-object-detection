import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Classification Training")

    # ===== DATA =====
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=128)

    parser.add_argument('--train_dir', type=str, default='data/classification/images/train')
    parser.add_argument('--val_dir', type=str, default='data/classification/images/val')

    parser.add_argument('--csv_dir', type=str, default='./data/CSVs')
    parser.add_argument('--out_dir', type=str, default='./outputs')

    # ===== TRAINING =====
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=15)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)

    return parser.parse_args()