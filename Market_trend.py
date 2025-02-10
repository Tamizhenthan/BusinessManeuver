import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_dataset():
    """Load a dataset from a CSV file."""
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    if not file_path:
        return None

    try:
        data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Dataset loaded successfully!")
        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
        return None

def visualize_data(data):
    """Visualize the loaded dataset with multiple plots."""
    try:
        # Create a new figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Plot 1: Histogram for the first numeric column
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            axes[0].hist(data[numeric_cols[0]], bins=10, color='blue', edgecolor='black')
            axes[0].set_title(f'Histogram of {numeric_cols[0]}')
            axes[0].set_xlabel(numeric_cols[0])
            axes[0].set_ylabel('Frequency')

        # Plot 2: Line graph for the first two numeric columns (if available)
        if len(numeric_cols) > 1:
            axes[1].plot(data[numeric_cols[0]], data[numeric_cols[1]], marker='o', linestyle='-')
            axes[1].set_title(f'Line Graph: {numeric_cols[0]} vs {numeric_cols[1]}')
            axes[1].set_xlabel(numeric_cols[0])
            axes[1].set_ylabel(numeric_cols[1])

        # Plot 3: Pie chart for a categorical column (if available)
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            category_counts = data[categorical_cols[0]].value_counts()
            axes[2].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
            axes[2].set_title(f'Pie Chart of {categorical_cols[0]}')

        # Plot 4: Scatter plot for the first two numeric columns (if available)
        if len(numeric_cols) > 1:
            axes[3].scatter(data[numeric_cols[0]], data[numeric_cols[1]], color='green')
            axes[3].set_title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
            axes[3].set_xlabel(numeric_cols[0])
            axes[3].set_ylabel(numeric_cols[1])

        # Adjust layout
        plt.tight_layout()

        # Create a canvas to display the figure in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Error", f"Visualization failed: {str(e)}")

def on_load_and_visualize():
    """Load dataset and visualize it."""
    data = load_dataset()
    if data is not None:
        visualize_data(data)

# Main Application
root = tk.Tk()
root.geometry("800x600")
root.title("Market Trend Visualization")

label = tk.Label(root, text="Market Trend Visualization", font=("Arial", 25))
label.pack(pady=20)

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

load_button = tk.Button(root, text="Load Dataset", command=on_load_and_visualize, font=("Arial", 12))
load_button.pack(pady=10)

root.mainloop()