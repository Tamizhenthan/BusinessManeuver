# import tkinter as tk
# import random
# from tkinter import filedialog, messagebox, ttk
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import seaborn as sns
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import numpy as np

# # Global variables
# loaded_data = None
# Business_Field = None

# def preprocess_data(file_path):
#     try:
#         # Read the dataset
#         data = pd.read_csv(file_path)

#         # Rename columns for consistency
#         data.rename(columns={
#             'Spending Score (1-100)': 'Spending_Score',
#             'Annual Income (k$)': 'Annual_Income'
#         }, inplace=True)

#         # Check for required columns
#         required_columns = ['Age', 'Spending_Score', 'Annual_Income']
#         for col in required_columns:
#             if col not in data.columns:
#                 raise ValueError(f"Dataset does not contain required column: {col}")

#         return data
#     except Exception as e:
#         raise ValueError(f"Error in loading or preprocessing data: {str(e)}")

# def load_dataset(root):
#     global loaded_data, Business_Field
#     try:
#         file_path = filedialog.askopenfilename(
#             filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
#         )
#         if not file_path:
#             return

#         loaded_data = preprocess_data(file_path)
#         messagebox.showinfo("Success", "Dataset loaded successfully!")

#         # Hide the "Load Dataset" button and dropdown after the dataset is loaded
#         root.load_button.pack_forget()
#         root.business_field_dropdown.pack_forget()

#         # Show the "Start Visualization" button after dataset is loaded
#         root.visualize_button.pack(pady=10)

#         # Show the "Dataset Loaded" label
#         if hasattr(root, "dataset_label"):
#             root.dataset_label.pack_forget()

#         root.dataset_label = tk.Label(root, text="Dataset Loaded", font=("Arial", 12), fg="green")
#         root.dataset_label.pack()

#         # Store the selected business field in the global variable
#         Business_Field = root.business_field_var.get()

#     except Exception as e:
#         messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

# def refresh_output(parent_frame, root):
#     try:
#         # Clear the previous output
#         for widget in parent_frame.winfo_children():
#             widget.destroy()

#         # Show the "Load Dataset" button and dropdown again after clearing
#         root.load_button.pack(pady=10)
#         root.business_field_dropdown.pack(pady=10)

#         # Hide the "Start Visualization" button after refresh
#         if hasattr(root, "visualize_button"):
#             root.visualize_button.pack_forget()

#         # Hide the "Dataset Loaded" label after refresh
#         if hasattr(root, "dataset_label"):
#             root.dataset_label.pack_forget()

#         # Hide the "Refresh" button after refresh
#         if hasattr(root, "refresh_button"):
#             root.refresh_button.pack_forget()

#         # Hide the "Generate Recommendation" button after refresh
#         if hasattr(root, "recommendation_button"):
#             root.recommendation_button.pack_forget()

#         # Hide the recommendation label after refresh
#         if hasattr(root, "recommendation_label"):
#             root.recommendation_label.pack_forget()

#     except AttributeError:
#         messagebox.showerror("Error", "Parent frame is not properly initialized.")

# def load_and_visualize_with_refresh(parent_frame, root):
#     global loaded_data
#     if loaded_data is None:
#         messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
#         return
#     refresh_output(parent_frame, root)
#     visualize_data(loaded_data, parent_frame, root)

# def visualize_data(data, parent_frame, root):
#     try:
#         # Hide "Dataset Loaded" label and "Start Visualization" button
#         if hasattr(root, "dataset_label"):
#             root.dataset_label.pack_forget()
#         if hasattr(root, "visualize_button"):
#             root.visualize_button.pack_forget()

#         # Hide "Load Dataset" button and dropdown
#         if hasattr(root, "load_button"):
#             root.load_button.pack_forget()
#         if hasattr(root, "business_field_dropdown"):
#             root.business_field_dropdown.pack_forget()

#         # Add title
#         title_label = tk.Label(parent_frame, text="Visualization Successful!", font=("Arial", 16), fg="green")
#         title_label.pack(pady=10)

#         # Add separator line
#         separator = ttk.Separator(parent_frame, orient="horizontal")
#         separator.pack(fill="x", pady=5)

#         # Create a scrollable frame
#         canvas = tk.Canvas(parent_frame)
#         scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
#         scrollable_frame = ttk.Frame(canvas)

#         scrollable_frame.bind(
#             "<Configure>",
#             lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
#         )

#         canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
#         canvas.configure(yscrollcommand=scrollbar.set)

#         canvas.pack(side="left", fill="both", expand=True)
#         scrollbar.pack(side="right", fill="y")

#         # Generate visualizations
#         fig, axes = plt.subplots(4, 3, figsize=(15, 20))
#         axes = axes.flatten()

#         # Plot 1: Age vs Spending Score
#         axes[0].scatter(data['Age'], data['Spending_Score'], c='blue')
#         axes[0].set_title('Age vs Spending Score')
#         axes[0].set_xlabel('Age')
#         axes[0].set_ylabel('Spending Score')
#         axes[0].grid(True)

#         # Plot 2: Annual Income vs Spending Score
#         axes[1].scatter(data['Annual_Income'], data['Spending_Score'], c='green')
#         axes[1].set_title('Annual Income vs Spending Score')
#         axes[1].set_xlabel('Annual Income (k$)')
#         axes[1].set_ylabel('Spending Score')
#         axes[1].grid(True)

#         # Plot 3: Age Distribution
#         axes[2].hist(data['Age'], bins=10, color='purple', edgecolor='black')
#         axes[2].set_title('Age Distribution')
#         axes[2].set_xlabel('Age')
#         axes[2].set_ylabel('Frequency')
#         axes[2].grid(axis='y')

#         # Plot 4: Annual Income Distribution
#         axes[3].hist(data['Annual_Income'], bins=10, color='orange', edgecolor='black')
#         axes[3].set_title('Annual Income Distribution')
#         axes[3].set_xlabel('Annual Income (k$)')
#         axes[3].set_ylabel('Frequency')
#         axes[3].grid(axis='y')

#         # Plot 5: Spending Score Distribution
#         axes[4].hist(data['Spending_Score'], bins=10, color='cyan', edgecolor='black')
#         axes[4].set_title('Spending Score Distribution')
#         axes[4].set_xlabel('Spending Score')
#         axes[4].set_ylabel('Frequency')
#         axes[4].grid(axis='y')

#         # Plot 6: Gender Distribution
#         gender_counts = data['Gender'].value_counts()
#         axes[5].bar(gender_counts.index, gender_counts.values, color=['pink', 'lightblue'])
#         axes[5].set_title('Gender Distribution')
#         axes[5].set_xlabel('Gender')
#         axes[5].set_ylabel('Count')
#         axes[5].grid(axis='y')

#         # Plot 7: Boxplot of Age
#         axes[6].boxplot(data['Age'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
#         axes[6].set_title('Boxplot of Age')
#         axes[6].set_xlabel('Age')

#         # Plot 8: Boxplot of Annual Income
#         axes[7].boxplot(data['Annual_Income'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
#         axes[7].set_title('Boxplot of Annual Income')
#         axes[7].set_xlabel('Annual Income (k$)')

#         # Plot 9: Boxplot of Spending Score
#         axes[8].boxplot(data['Spending_Score'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
#         axes[8].set_title('Boxplot of Spending Score')
#         axes[8].set_xlabel('Spending Score')

#         # Plot 10: Heatmap of Correlations
#         correlation_matrix = data[['Age', 'Annual_Income', 'Spending_Score']].corr()
#         sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[9])
#         axes[9].set_title('Heatmap of Correlations')

#         # Plot 11: Pie Chart of Clusters
#         kmeans = KMeans(n_clusters=5, random_state=42)
#         data['Cluster'] = kmeans.fit_predict(data[['Annual_Income', 'Spending_Score']])
#         cluster_counts = data['Cluster'].value_counts()
#         axes[10].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
#                      autopct='%1.1f%%', colors=['red', 'blue', 'green', 'purple', 'orange'])
#         axes[10].set_title('Cluster Distribution')

#         # Adjust layout and add to canvas
#         fig.tight_layout()
#         canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
#         canvas_fig.get_tk_widget().pack(fill="both", expand=True)

#         # Plot 12: K-Means Clustering (integrate with Tkinter)
#         cluster_fig, cluster_ax = plt.subplots(figsize=(10, 8))
#         for cluster in range(5):
#             cluster_data = data[data['Cluster'] == cluster]
#             cluster_ax.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'],
#                                label=f'Cluster {cluster}')
#         cluster_ax.set_title('K-Means Clustering')
#         cluster_ax.set_xlabel('Annual Income (k$)')
#         cluster_ax.set_ylabel('Spending Score')
#         cluster_ax.legend()

#         canvas_cluster = FigureCanvasTkAgg(cluster_fig, master=scrollable_frame)
#         canvas_cluster.get_tk_widget().pack(fill="both", expand=True)

#         # Plot 13: Pair Plot
#         pairplot_fig = sns.pairplot(data[['Age', 'Annual_Income', 'Spending_Score']])
#         pairplot_canvas = FigureCanvasTkAgg(pairplot_fig.fig, master=scrollable_frame)
#         pairplot_canvas.get_tk_widget().pack(fill="both", expand=True)

#         # Show Refresh button
#         show_refresh_button(root)

#         # Show Generate Recommendation button
#         show_recommendation_button(root)

#     except Exception as e:
#         messagebox.showerror("Error", f"Visualization failed: {str(e)}")

# def show_refresh_button(root):
#     if not hasattr(root, "refresh_button"):
#         refresh_button = tk.Button(
#             root,
#             text="Refresh",
#             command=lambda: refresh_output(frame, root),
#             font=("Arial", 12),
#             fg="white",
#             bg="red",
#         )
#         root.refresh_button = refresh_button
#     root.refresh_button.pack(pady=10)

# def show_recommendation_button(root):
#     if not hasattr(root, "recommendation_button"):
#         recommendation_button = tk.Button(
#             root,
#             text="Generate Recommendation",
#             command=lambda: generate_recommendation(root),
#             font=("Arial", 12),
#             fg="white",
#             bg="blue",
#         )
#         root.recommendation_button = recommendation_button
#     root.recommendation_button.pack(pady=10)


# def generate_recommendation(root):
#     global Business_Field, loaded_data

#     if loaded_data is None:
#         messagebox.showerror("Error", "Dataset is not loaded. Please load the dataset.")
#         return

#     # Analyze dataset
#     income_mean = loaded_data['Annual_Income'].mean()
#     income_75th = np.percentile(loaded_data['Annual_Income'], 75)
#     income_25th = np.percentile(loaded_data['Annual_Income'], 25)
#     high_income_count = sum(loaded_data['Annual_Income'] > 150)  # Count customers with income > 150k

#     spending_mean = loaded_data['Spending_Score'].mean()
#     spending_75th = np.percentile(loaded_data['Spending_Score'], 75)
#     spending_25th = np.percentile(loaded_data['Spending_Score'], 25)

#     age_mean = loaded_data['Age'].mean()
#     age_median = loaded_data['Age'].median()
#     gender_counts = loaded_data['Gender'].value_counts()

#     male_percentage = (gender_counts.get("Male", 0) / len(loaded_data)) * 100
#     female_percentage = (gender_counts.get("Female", 0) / len(loaded_data)) * 100

#     # Define recommendations based on segmentation and market trends
#     recommendations = {
#         "Medicine": [
#             f"With {female_percentage:.1f}% of customers being female, consider launching women-specific health packages like maternity care or hormonal health supplements.",
#             f"Approximately 25% of customers earn below ${income_25th:.1f}k annually. Focus on affordable medicines or subscription plans to cater to this segment.",
#             f"Only {high_income_count} customers earn more than $150k. Launch premium health packages but in limited quantities to minimize financial risk.",
#             f"With an average spending score of {spending_mean:.1f}, customers are moderately engaged. Increase engagement with loyalty programs or free health checkups.",
#             f"Customers aged around {age_median:.1f} are a significant demographic. Introduce midlife wellness products like heart care supplements."
#         ],
#         "Automobiles": [
#             f"High-income earners (above ${income_75th:.1f}k) make up only {len(loaded_data[loaded_data['Annual_Income'] > income_75th])} customers. Target this segment with premium models but maintain a small production scale to reduce risks.",
#             f"Customers earning below ${income_25th:.1f}k form {len(loaded_data[loaded_data['Annual_Income'] < income_25th])} of your base. Offer affordable, fuel-efficient vehicles to expand your reach.",
#             f"The median spending score is {spending_mean:.1f}. Focus on mid-range models with financing options to appeal to moderate spenders.",
#             f"Young customers (median age: {age_median:.1f}) may prefer sporty, compact cars. Launch campaigns targeting this age group through social media.",
#             f"Electric vehicles are trending in the market with a 30% increase in adoption globally. Introduce an entry-level EV to attract eco-conscious buyers."
#         ],
#         "Clothing": [
#             f"The data shows {female_percentage:.1f}% of your customers are female. Expand your women’s clothing line, emphasizing stylish but affordable options.",
#             f"Customers earning more than ${income_75th:.1f}k make up only {len(loaded_data[loaded_data['Annual_Income'] > income_75th])} people. Launch a premium clothing range but in smaller quantities to test demand.",
#             f"The median annual income is ${income_mean:.1f}k, suggesting a preference for budget-friendly, casual wear. Focus on mass-market collections with seasonal discounts.",
#             f"Customers aged {age_mean:.1f} are likely to respond well to youth-oriented trends. Collaborate with influencers to create buzz for your new collections.",
#             f"Current market trends show a rise in sustainable fashion by 20% this year. Introduce eco-friendly materials to attract environmentally conscious buyers."
#         ],
#         "Footwear": [
#             f"The average income of ${income_mean:.1f}k suggests a focus on durable, affordable footwear. Highlight value-for-money designs in your promotions.",
#             f"With a spending score average of {spending_mean:.1f}, promote mid-range or discount footwear collections to increase conversion.",
#             f"Only {high_income_count} high-income customers (above $150k) exist in your dataset. Introduce premium footwear in small quantities to limit risk.",
#             f"Customers aged around {age_median:.1f} might prefer comfort-focused footwear. Introduce orthotic-friendly options to meet this demand.",
#             f"With 15% of the global market favoring sports shoes, target young customers (median age: {age_median:.1f}) with trendy, athletic designs."
#         ],
#         "Electronics": [
#             f"Your data shows only {high_income_count} high-income customers. Introduce premium smart home devices but limit production to gauge demand.",
#             f"The average income of ${income_mean:.1f}k suggests promoting mid-range electronic devices with financing or installment plans.",
#             f"Customers with spending scores above {spending_75th:.1f} are prime targets for bundled product deals or extended warranties.",
#             f"Young customers (median age: {age_median:.1f}) might be interested in gadgets like wearables, gaming devices, or Bluetooth speakers.",
#             f"The market for AI-powered home assistants grew by 25% last year. Introduce entry-level smart devices to capture this trend."
#         ],
#         "Gadgets": [
#             f"Young customers (median age: {age_median:.1f}) are likely to prefer wearable gadgets. Launch budget-friendly smartwatches or fitness bands.",
#             f"Only {len(loaded_data[loaded_data['Annual_Income'] > 150])} customers earn over $150k. Test premium gadget launches in small batches to avoid overproduction.",
#             f"Customers with spending scores below {spending_25th:.1f} might respond well to discounts or trade-in offers for older gadgets.",
#             f"Male customers ({male_percentage:.1f}%) are prime targets for gaming accessories or high-performance tech products.",
#             f"Global trends show a 35% increase in wireless earbuds sales. Target younger buyers with competitive pricing and promotions."
#         ]
#     }

#     # Randomly select 2-3 recommendations
#     selected_recommendations = random.sample(recommendations.get(Business_Field, ["No recommendation available."]), k=min(3, len(recommendations.get(Business_Field, []))))

#     # Display recommendations
#     recommendation_text = "\n\n".join(selected_recommendations)

#     if hasattr(root, "recommendation_label"):
#         root.recommendation_label.pack_forget()

#     root.recommendation_label = tk.Label(
#         root,
#         text=recommendation_text,
#         font=("Arial", 14),
#         fg="black",
#         wraplength=600,
#         justify="center"
#     )
#     root.recommendation_label.pack(pady=10)




# # Main Application
# root = tk.Tk()
# root.geometry("800x600")

# label = tk.Label(root, text="Business Maneuver", font=("Arial", 25))
# label.pack(pady=20)

# frame = ttk.Frame(root)
# frame.pack(fill="both", expand=True)

# root.frame = frame  # Attach the frame to root for proper reference

# root.business_field_var = tk.StringVar(value="Select Business Field")
# root.business_field_dropdown = ttk.Combobox(
#     root,
#     textvariable=root.business_field_var,
#     values=["Medicine", "Automobiles", "Clothing", "Footwear", "Electronics", "Gadgets"],
#     state="readonly",
#     font=("Arial", 12)
# )
# root.business_field_dropdown.pack(pady=10)

# root.load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset(root), font=("Arial", 12))
# root.load_button.pack(pady=10)

# root.visualize_button = tk.Button(root, text="Start Visualization", command=lambda: load_and_visualize_with_refresh(frame, root), font=("Arial", 12))

# root.mainloop()

import tkinter as tk
import random
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Global variables
loaded_data = None
Business_Field = None

def preprocess_data(file_path):
    try:
        data = pd.read_csv(file_path)

        # Drop columns like 'id', 'ID', or any similar column that doesn't make sense for visualization
        id_columns = [col for col in data.columns if 'id' in col.lower()]
        data.drop(columns=id_columns, inplace=True)

        return data
    except Exception as e:
        raise ValueError(f"Error in loading or preprocessing data: {str(e)}")

def load_dataset(root):
    global loaded_data, Business_Field
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not file_path:
            return

        loaded_data = preprocess_data(file_path)
        messagebox.showinfo("Success", "Dataset loaded successfully!")

        root.load_button.pack_forget()
        root.business_field_dropdown.pack_forget()
        root.visualize_button.pack(pady=10)

        if hasattr(root, "dataset_label"):
            root.dataset_label.pack_forget()

        root.dataset_label = tk.Label(root, text="Dataset Loaded", font=("Arial", 12), fg="green")
        root.dataset_label.pack()

        Business_Field = root.business_field_var.get()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

def refresh_output(parent_frame, root):
    try:
        for widget in parent_frame.winfo_children():
            widget.destroy()

        root.load_button.pack(pady=10)
        root.business_field_dropdown.pack(pady=10)
        root.visualize_button.pack_forget()

        if hasattr(root, "dataset_label"):
            root.dataset_label.pack_forget()

        if hasattr(root, "refresh_button"):
            root.refresh_button.pack_forget()

        if hasattr(root, "recommendation_button"):
            root.recommendation_button.pack_forget()

        if hasattr(root, "recommendation_label"):
            root.recommendation_label.pack_forget()

    except AttributeError:
        messagebox.showerror("Error", "Parent frame is not properly initialized.")

def load_and_visualize_with_refresh(parent_frame, root):
    global loaded_data
    if loaded_data is None:
        messagebox.showerror("Error", "No dataset loaded. Please load a dataset first.")
        return
    refresh_output(parent_frame, root)
    visualize_data(loaded_data, parent_frame, root)

def visualize_data(data, parent_frame, root):
    try:
        if hasattr(root, "dataset_label"):
            root.dataset_label.pack_forget()
        if hasattr(root, "visualize_button"):
            root.visualize_button.pack_forget()

        if hasattr(root, "load_button"):
            root.load_button.pack_forget()
        if hasattr(root, "business_field_dropdown"):
            root.business_field_dropdown.pack_forget()

        notebook = ttk.Notebook(parent_frame)
        notebook.pack(fill="both", expand=True)

        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        tab3 = ttk.Frame(notebook)

        notebook.add(tab1, text="Histograms & Pie Charts")
        notebook.add(tab2, text="Scatter & Line Plots")
        notebook.add(tab3, text="K-Means Clustering")

        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        # Histograms and Pie Charts
        fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))

        if len(numeric_cols) > 0:
            axes1[0].hist(data[numeric_cols[0]], bins=10, color='blue', edgecolor='black')
            axes1[0].set_title(f'Histogram of {numeric_cols[0]}')

        if len(categorical_cols) > 0:
            category_counts = data[categorical_cols[0]].value_counts()
            axes1[1].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
            axes1[1].set_title(f'Pie Chart of {categorical_cols[0]}')

        canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
        canvas1.get_tk_widget().pack(fill="both", expand=True)

        # Scatter and Line Plots
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

        if len(numeric_cols) > 1:
            axes2[0].scatter(data[numeric_cols[0]], data[numeric_cols[1]], color='green')
            axes2[0].set_title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')

            axes2[1].plot(data[numeric_cols[0]], data[numeric_cols[1]], marker='o', linestyle='-')
            axes2[1].set_title(f'Line Graph: {numeric_cols[0]} vs {numeric_cols[1]}')

        canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
        canvas2.get_tk_widget().pack(fill="both", expand=True)

        # K-Means Clustering
        fig3, axes3 = plt.subplots(1, 2, figsize=(10, 5))

        if len(numeric_cols) >= 2:
            kmeans = KMeans(n_clusters=3, random_state=42)
            data['Cluster'] = kmeans.fit_predict(data[numeric_cols[:2]])

            axes3[0].scatter(data[numeric_cols[0]], data[numeric_cols[1]], c=data['Cluster'], cmap='viridis')
            axes3[0].set_title('K-Means Clustering')

            cluster_counts = data['Cluster'].value_counts()
            axes3[1].pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
                         autopct='%1.1f%%', startangle=90)
            axes3[1].set_title('Cluster Distribution')

        canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
        canvas3.get_tk_widget().pack(fill="both", expand=True)

        show_refresh_button(root)

    except Exception as e:
        messagebox.showerror("Error", f"Visualization failed: {str(e)}")

def show_refresh_button(root):
    if not hasattr(root, "refresh_button"):
        refresh_button = tk.Button(root, text="Refresh", command=lambda: refresh_output(frame, root),
                                   font=("Arial", 12), fg="white", bg="red")
        root.refresh_button = refresh_button
    root.refresh_button.pack(pady=10)

# Main Application
root = tk.Tk()
root.geometry("900x700")

label = tk.Label(root, text="Business Maneuver", font=("Arial", 25))
label.pack(pady=20)

frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)

root.frame = frame  

root.business_field_var = tk.StringVar(value="Select Business Field")
root.business_field_dropdown = ttk.Combobox(root, textvariable=root.business_field_var,
                                            values=["Medicine", "Automobiles", "Clothing", "Footwear", "Electronics", "Gadgets"],
                                            state="readonly", font=("Arial", 12))
root.business_field_dropdown.pack(pady=10)

root.load_button = tk.Button(root, text="Load Dataset", command=lambda: load_dataset(root), font=("Arial", 12))
root.load_button.pack(pady=10)

root.visualize_button = tk.Button(root, text="Start Visualization", command=lambda: load_and_visualize_with_refresh(frame, root), font=("Arial", 12))

root.mainloop()
