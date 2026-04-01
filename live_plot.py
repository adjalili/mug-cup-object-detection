import pandas as pd
import matplotlib.pyplot as plt
import time

def plot_live():
    plt.ion() # Interactive mode on කිරීම
    fig, ax = plt.subplots()

    while True:
        try:
            data = pd.read_csv("data/CSVs/training_log.csv") # මෙතැන path එක නිවැරදි කරන්න
            ax.clear()
            ax.plot(data['epoch'], data['train_loss'], label='Train Loss', color='blue')
            ax.plot(data['epoch'], data['val_loss'], label='Val Loss', color='red')
            
            ax.set_title("Live Learning Curve")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(10) # තත්පර 10කට වරක් update වීම
        except Exception as e:
            print("Waiting for data...")
            time.sleep(5)

if __name__ == "__main__":
    plot_live()