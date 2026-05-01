import argparse




def get_args():
    parser = argparse.ArgumentParser(description="Model training.")
    
    parser.add_argument('--backbone', type=str, default='FasterRCNN_resnet50_fpn',
                        choices=['FasterRCNN_resnet50_fpn'])
    
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)

    parser.add_argument('--csv_dir', type=str, default='data/CSVs')
    parser.add_argument('--output_dir', type=str, default='./sessions')

    parser.add_argument('--batch_size', type=int, default=4,
                        choices= [4, 8, 16, 32, 64])
    
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=1e-4)

    return parser.parse_args()
    