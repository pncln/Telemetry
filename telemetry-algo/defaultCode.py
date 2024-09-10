import pandas as pd  # Pandas for data manipulation

import matplotlib
import matplotlib.pyplot as plt  # Matplotlib for data visualization
import sys
import os
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np  # NumPy for numerical operations
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
import seaborn as sns  # Seaborn for data visualization

# Import machine learning models
from sklearn.naive_bayes import GaussianNB, BernoulliNB  # For binary classification
from sklearn.naive_bayes import MultinomialNB  # For multi-class classification
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting classifier


# Import machine learning evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Import tools for data splitting
from sklearn.model_selection import train_test_split

figures = []

if getattr(sys, 'frozen', False):
    print("Running in a PyInstaller bundle")
    print("os.path.dirname(sys.executable):", os.path.dirname(sys.executable))
else:
    print("Running in a normal Python environment")
    print("os.path.dirname(sys.executable)):", os.path.dirname(os.path.abspath(__file__)))

print("Current working directory:", os.getcwd())


def transform_csv(input_file, output_file):
    data = []
    current_device = ""

    with open(input_file, 'r') as file:
        for line in file:
            if line.startswith("Mnemonic:"):
                current_device = line.split(":")[1].strip()
            elif line.startswith("X Time"):
                continue
            elif "," in line:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    date = parts[0].strip()
                    metric1 = parts[1].strip()
                    data.append([date, current_device, 0, metric1])

    df = pd.DataFrame(data, columns=['date', 'device', 'failure', 'metric1'])
    df.to_csv(output_file, index=False)

transform_csv('input_dataset.csv', 'output_dataset.csv')

# Determine the real path to the executable or script
if getattr(sys, 'frozen', False):  # Check if the app is frozen (i.e., packaged with PyInstaller)
    base_path = os.path.dirname(sys.executable)  # Path of the actual executable
else:
    base_path = os.path.dirname(os.path.abspath(__file__))  # Path where the script is located

# Use this base_path to construct the path to your CSV file or other resources
filename = os.path.join(base_path, 'output_dataset.csv')
print("Path to the CSV file:", os.path.join(base_path, 'output_dataset.csv'))
df = pd.read_csv(filename)
num_unique_devices = df['device'].nunique()

# Remove zero-valued metric1 data
df = df[df['metric1'] != 0]

matplotlib.use('QtAgg')

df.shape
df.drop_duplicates(inplace=True)
df.shape

def summarize_data(df):
    print("Number of rows and columns:", df.shape, flush=True)
    print("\nColumns in the dataset:", df.columns, flush=True)
    print("\nData types and missing values:", flush=True)
    print(df.info(), flush=True)
    print("\nSummary statistics for numerical columns:", flush=True)
    print(df.describe(), flush=True)
    print("\nMissing values:", flush=True)
    print(df.isnull().sum(), flush=True)
    print("\nUnique values in 'failure' column:", flush=True)
    print(df['failure'].value_counts(), flush=True)
    print(f"Number of unique devices: {num_unique_devices}", flush=True)

summarize_data(df)

def plot_metric1_by_device(df):
    devices = df['device'].unique()
    num_devices = len(devices)
    
    fig, axes = plt.subplots(num_devices, 1, figsize=(12, 5*num_devices), sharex=True)
    fig.suptitle('', fontsize=16)
    
    for i, device in enumerate(devices):
        device_data = df[df['device'] == device]
        axes[i].plot(pd.to_datetime(device_data['date']), device_data['metric1'])
        axes[i].set_title(f'Device: {device}')
        axes[i].set_ylabel('[W]')
    
    plt.xlabel('Date')
    plt.tight_layout()
    figures.append(fig)

# Call the function after reading the CSV
plot_metric1_by_device(df)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Predictive Maintenance Analysis")
        self.setGeometry(100, 100, 1000, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        print(f"Number of unique figures: {len(figures)}", flush=True)

        for i, fig in enumerate(figures):
            # Reduce font size for all text elements in the figure
            for ax in fig.axes:
                for item in ([fig._suptitle] + 
                            [ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                    if item:
                        item.set_fontsize(8)  # Adjust this value as needed

            # If there's a legend, reduce its font size too
            legend = fig.legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(6)  # Adjust this value as needed

            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            tab = QWidget()
            tab_layout = QVBoxLayout(tab)

            toolbar = NavigationToolbar(canvas, self)
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)

            self.tab_widget.addTab(tab, f"Plot {i+1}")

        self.tab_widget.currentChanged.connect(self.resize_plot)

    def resize_plot(self, index):
        current_tab = self.tab_widget.widget(index)
        canvas = current_tab.layout().itemAt(1).widget()
        fig = canvas.figure
        fig.tight_layout()
        canvas.draw()

class PredictiveMaintenanceWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        self.main_window = MainWindow()
        layout.addWidget(self.main_window.tab_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Set a default style
    font = QFont("Arial", 14)  # Set a default font
    app.setFont(font)
    main_window = PredictiveMaintenanceWidget()
    main_window.show()
    sys.exit(app.exec())