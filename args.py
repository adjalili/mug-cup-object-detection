import argparse


def get_args():
    parser = argparse.ArgumentParser(description='model training options')

    parser.add_argument('--backbone', type=str, default="fasterrcnn_resnet50_fpn",
                        choices=["fasterrcnn_resnet50_fpn", 'fasterrcnn_mobilenet_v3'])

    parser.add_argument('--csv_dir', type= str, default="./data/CSVs")
    parser.add_argument('--out_dir', type=str, default="./data/CSVs")

    parser.add_argument('--batch_size', type=int, default=2,
                        choices=[8, 16, 32, 64])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    return parser.parse_args()
