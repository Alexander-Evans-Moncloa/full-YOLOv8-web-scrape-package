import matplotlib.pyplot as plt
import pandas as pd

#Displaying the results
results_path = '/home/alexander/runs/classify/train11/results.csv'  #Update with your correct path

#Read the CSV file into a pandas DataFrame
results = pd.read_csv(results_path)

#Ensure that column names are stripped of leading/trailing spaces
results.columns = results.columns.str.strip()

#Convert columns to numpy arrays to avoid multi-dimensional indexing errors
epochs = results['epoch'].to_numpy()
train_loss = results['train/loss'].to_numpy()
val_loss = results['val/loss'].to_numpy()
accuracy = (results['metrics/accuracy_top1'].to_numpy() * 100)

#Plotting the results using the arrays
plt.figure()  # Create a new figure

#Plot 'train/loss' vs 'epoch'
plt.plot(epochs, train_loss, label = 'train/loss')

#Plot 'val/loss' vs 'epoch'
plt.plot(epochs, val_loss, label = 'val/loss', color = 'red')

#Set up grid, title, and labels
plt.grid()
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#Show legend
plt.legend()

#Plotting the results using the arrays
plt.figure()  #Create a new figure

#Plot 'accuracy' vs 'epoch'
plt.plot(epochs, accuracy)

#Set up grid, title, and labels
plt.grid()
plt.title('Validation Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')

#Show legend
plt.legend()


# Display all the plots
plt.show()
