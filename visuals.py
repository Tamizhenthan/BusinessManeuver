# visuals.py

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox

def preprocess_data(file_path):
    try:
        # Read the dataset
        data = pd.read_csv(file_path)

        # Rename columns for consistency
        data.rename(columns={
            'Spending Score (1-100)': 'Spending_Score',
            'Annual Income (k$)': 'Annual_Income'
        }, inplace=True)

        # Check for required columns
        required_columns = ['Age', 'Spending_Score', 'Annual_Income']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Dataset does not contain required column: {col}")

        return data
    except Exception as e:
        raise ValueError(f"Error in loading or preprocessing data: {str(e)}")

def visualize_data(data, parent_frame):
    try:
        # Generate visualizations
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        axes = axes.flatten()

        # Plot 1: Age vs Spending Score
        axes[0].scatter(data['Age'], data['Spending_Score'], c='blue')
        axes[0].set_title('Age vs Spending Score')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Spending Score')
        axes[0].grid(True)

        # Plot 2: Annual Income vs Spending Score
        axes[1].scatter(data['Annual_Income'], data['Spending_Score'], c='green')
        axes[1].set_title('Annual Income vs Spending Score')
        axes[1].set_xlabel('Annual Income (k$)')
        axes[1].set_ylabel('Spending Score')
        axes[1].grid(True)

        # Plot 3: Age Distribution
        axes[2].hist(data['Age'], bins=10, color='purple', edgecolor='black')
        axes[2].set_title('Age Distribution')
        axes[2].set_xlabel('Age')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(axis='y')

        # Plot 4: Annual Income Distribution
        axes[3].hist(data['Annual_Income'], bins=10, color='orange', edgecolor='black')
        axes[3].set_title('Annual Income Distribution')
        axes[3].set_xlabel('Annual Income (k$)')
        axes[3].set_ylabel('Frequency')
        axes[3].grid(axis='y')

        # Plot 5: Spending Score Distribution
        axes[4].hist(data['Spending_Score'], bins=10, color='cyan', edgecolor='black')
        axes[4].set_title('Spending Score Distribution')
        axes[4].set_xlabel('Spending Score')
        axes[4].set_ylabel('Frequency')
        axes[4].grid(axis='y')

        # Plot 6: Gender Distribution
        gender_counts = data['Gender'].value_counts()
        axes[5].bar(gender_counts.index, gender_counts.values, color=['pink', 'lightblue'])
        axes[5].set_title('Gender Distribution')
        axes[5].set_xlabel('Gender')
        axes[5].set_ylabel('Count')
        axes[5].grid(axis='y')

        # Plot 7: Boxplot of Age
        axes[6].boxplot(data['Age'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes[6].set_title('Boxplot of Age')
        axes[6].set_xlabel('Age')

        # Plot 8: Boxplot of Annual Income
        axes[7].boxplot(data['Annual_Income'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        axes[7].set_title('Boxplot of Annual Income')
        axes[7].set_xlabel('Annual Income (k$)')

        # Plot 9: Boxplot of Spending Score
        axes[8].boxplot(data['Spending_Score'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
        axes[8].set_title('Boxplot of Spending Score')
        axes[8].set_xlabel('Spending Score')

        # Plot 10: Heatmap of Correlations
        correlation_matrix = data[['Age', 'Annual_Income', 'Spending_Score']].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[9])
        axes[9].set_title('Heatmap of Correlations')

        # Plot 11: Pie Chart of Clusters
        kmeans = KMeans(n_clusters=5, random_state=42)
        data['Cluster'] = kmeans.fit_predict(data[['Annual_Income', 'Spending_Score']])
        cluster_counts = data['Cluster'].value_counts()
        axes[10].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                     autopct='%1.1f%%', colors=['red', 'blue', 'green', 'purple', 'orange'])
        axes[10].set_title('Cluster Distribution')

        # Adjust layout and add to canvas
        fig.tight_layout()
        canvas_fig = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_fig.get_tk_widget().pack(fill="both", expand=True)

        # Plot 12: K-Means Clustering
        cluster_fig, cluster_ax = plt.subplots(figsize=(10, 8))
        for cluster in range(5):
            cluster_data = data[data['Cluster'] == cluster]
            cluster_ax.scatter(cluster_data['Annual_Income'], cluster_data['Spending_Score'],
                               label=f'Cluster {cluster}')
        cluster_ax.set_title('K-Means Clustering')
        cluster_ax.set_xlabel('Annual Income (k$)')
        cluster_ax.set_ylabel('Spending Score')
        cluster_ax.legend()

        canvas_cluster = FigureCanvasTkAgg(cluster_fig, master=parent_frame)
        canvas_cluster.get_tk_widget().pack(fill="both", expand=True)

        # Plot 13: Pair Plot
        pairplot_fig = sns.pairplot(data[['Age', 'Annual_Income', 'Spending_Score']])
        pairplot_canvas = FigureCanvasTkAgg(pairplot_fig.fig, master=parent_frame)
        pairplot_canvas.get_tk_widget().pack(fill="both", expand=True)

    except Exception as e:
        messagebox.showerror("Error", f"Visualization failed: {str(e)}")

def generate_recommendation(loaded_data, business_field):
    if loaded_data is None:
        return "Dataset is not loaded. Please load the dataset."

    # Analyze dataset
    income_mean = loaded_data['Annual_Income'].mean()
    income_75th = np.percentile(loaded_data['Annual_Income'], 75)
    income_25th = np.percentile(loaded_data['Annual_Income'], 25)
    high_income_count = sum(loaded_data['Annual_Income'] > 150)  # Count customers with income > 150k

    spending_mean = loaded_data['Spending_Score'].mean()
    spending_75th = np.percentile(loaded_data['Spending_Score'], 75)
    spending_25th = np.percentile(loaded_data['Spending_Score'], 25)

    age_mean = loaded_data['Age'].mean()
    age_median = loaded_data['Age'].median()
    gender_counts = loaded_data['Gender'].value_counts()

    male_percentage = (gender_counts.get("Male", 0) / len(loaded_data)) * 100
    female_percentage = (gender_counts.get("Female", 0) / len(loaded_data)) * 100

    # Define recommendations based on segmentation and market trends
    recommendations = {
        "Medicine": [
            f"With {female_percentage:.1f}% of customers being female, consider launching women-specific health packages like maternity care or hormonal health supplements.",
            f"Approximately 25% of customers earn below ${income_25th:.1f}k annually. Focus on affordable medicines or subscription plans to cater to this segment.",
            f"Only {high_income_count} customers earn more than $150k. Launch premium health packages but in limited quantities to minimize financial risk.",
            f"With an average spending score of {spending_mean:.1f}, customers are moderately engaged. Increase engagement with loyalty programs or free health checkups.",
            f"Customers aged around {age_median:.1f} are a significant demographic. Introduce midlife wellness products like heart care supplements."
        ],
        "Automobiles": [
            f"High-income earners (above ${income_75th:.1f}k) make up only {len(loaded_data[loaded_data['Annual_Income'] > income_75th])} customers. Target this segment with premium models but maintain a small production scale to reduce risks.",
            f"Customers earning below ${income_25th:.1f}k form {len(loaded_data[loaded_data['Annual_Income'] < income_25th])} of your base. Offer affordable, fuel-efficient vehicles to expand your reach.",
            f"The median spending score is {spending_mean:.1f}. Focus on mid-range models with financing options to appeal to moderate spenders.",
            f"Young customers (median age: {age_median:.1f}) may prefer sporty, compact cars. Launch campaigns targeting this age group through social media.",
            f"Electric vehicles are trending in the market with a 30% increase in adoption globally. Introduce an entry-level EV to attract eco-conscious buyers."
        ],
        "Clothing": [
            f"The data shows {female_percentage:.1f}% of your customers are female. Expand your women’s clothing line, emphasizing stylish but affordable options.",
            f"Customers earning more than ${income_75th:.1f}k make up only {len(loaded_data[loaded_data['Annual_Income'] > income_75th])} people. Launch a premium clothing range but in smaller quantities to test demand.",
            f"The median annual income is ${income_mean:.1f}k, suggesting a preference for budget-friendly, casual wear. Focus on mass-market collections with seasonal discounts.",
            f"Customers aged {age_mean:.1f} are likely to respond well to youth-oriented trends. Collaborate with influencers to create buzz for your new collections.",
            f"Current market trends show a rise in sustainable fashion by 20% this year. Introduce eco-friendly materials to attract environmentally conscious buyers."
        ],
        "Footwear": [
            f"The average income of ${income_mean:.1f}k suggests a focus on durable, affordable footwear. Highlight value-for-money designs in your promotions.",
            f"With a spending score average of {spending_mean:.1f}, promote mid-range or discount footwear collections to increase conversion.",
            f"Only {high_income_count} high-income customers (above $150k) exist in your dataset. Introduce premium footwear in small quantities to limit risk.",
            f"Customers aged around {age_median:.1f} might prefer comfort-focused footwear. Introduce orthotic-friendly options to meet this demand.",
            f"With 15% of the global market favoring sports shoes, target young customers (median age: {age_median:.1f}) with trendy, athletic designs."
        ],
        "Electronics": [
            f"Your data shows only {high_income_count} high-income customers. Introduce premium smart home devices but limit production to gauge demand.",
            f"The average income of ${income_mean:.1f}k suggests promoting mid-range electronic devices with financing or installment plans.",
            f"Customers with spending scores above {spending_75th:.1f} are prime targets for bundled product deals or extended warranties.",
            f"Young customers (median age: {age_median:.1f}) might be interested in gadgets like wearables, gaming devices, or Bluetooth speakers.",
            f"The market for AI-powered home assistants grew by 25% last year. Introduce entry-level smart devices to capture this trend."
        ],
        "Gadgets": [
            f"Young customers (median age: {age_median:.1f}) are likely to prefer wearable gadgets. Launch budget-friendly smartwatches or fitness bands.",
            f"Only {len(loaded_data[loaded_data['Annual_Income'] > 150])} customers earn over $150k. Test premium gadget launches in small batches to avoid overproduction.",
            f"Customers with spending scores below {spending_25th:.1f} might respond well to discounts or trade-in offers for older gadgets.",
            f"Male customers ({male_percentage:.1f}%) are prime targets for gaming accessories or high-performance tech products.",
            f"Global trends show a 35% increase in wireless earbuds sales. Target younger buyers with competitive pricing and promotions."
        ]
    }

    # Randomly select 2-3 recommendations
    selected_recommendations = random.sample(recommendations.get(business_field, ["No recommendation available."]), k=min(3, len(recommendations.get(business_field, []))))

    return "\n\n".join(selected_recommendations)