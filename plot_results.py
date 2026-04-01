import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curve(csv_file):
    try:
        # CSV ගොනුව කියවීම
        data = pd.read_csv(csv_file)
        
        epochs = data['epoch']
        train_loss = data['train_loss']
        val_loss = data['val_loss']

        # ප්‍රස්ථාරය ඇඳීම
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Training Loss', marker='o', color='blue')
        plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='red')

        # විස්තර ඇතුළත් කිරීම
        plt.title('Learning Curve - Object Detection')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)

        # පින්තූරය save කිරීම
        plt.savefig('learning_curve.png')
        print("Success: Learning curve saved as 'learning_curve.png'")
        
        # ඔබට screen එකේ බැලීමට අවශ්‍ය නම් (Local PC එකේදී පමණයි)
        # plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_learning_curve('data/CSVs/training_log.csv')