import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV data
try:
    data = pd.read_csv('mcdonalds.csv')
    print("CSV file loaded successfully.")
except FileNotFoundError:
    print("Error: File 'mcdonalds.csv' not found.")
    exit()

# Check for required columns
required_columns = ['Gender', 'Age']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Missing required columns: {missing_columns}")
    exit()

# Create charts folder
os.makedirs("charts", exist_ok=True)

# 1. Simulated Pie Chart (Customer Segments)
plt.figure(figsize=(8, 6))
segments = ['Young Budget Eaters', 'Health Conscious', 'Fast Food Lovers', 'Family Focused', 'Premium Seekers']
sizes = [20, 15, 25, 20, 20]
colors = sns.color_palette('pastel', len(segments))
plt.pie(sizes, labels=segments, colors=colors, startangle=90, shadow=True, autopct='%1.1f%%')
plt.title("Customer Segments (Simulated)")
plt.tight_layout()
plt.savefig("charts/1_segments_pie.png")
plt.close()

# 2. Gender Donut Chart
gender_counts = data['Gender'].value_counts()
plt.figure(figsize=(6, 6))
colors = ['#ffcc99', '#66b3ff']
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90,
        colors=colors, wedgeprops={'width': 0.4})
plt.title("Gender Distribution")
plt.tight_layout()
plt.savefig("charts/2_gender_donut.png")
plt.close()

# 3. Histogram of Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data['Age'], bins=30, kde=False, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig("charts/3_age_histogram.png")
plt.close()

print("✅ Charts generated successfully (without income-based plot).")




