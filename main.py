# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'D:/SEM 7/SWMA/Project/Phase 1/blinkit.csv' 
df = pd.read_csv(file_path)

# Check the basic structure of the dataset
print("Data Head:\n", df.head())
print("\nData Info:\n", df.info())

# Data Cleaning and Preprocessing
# Convert purchase_date to datetime format
df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

# Remove any rows with missing or corrupted data
df.dropna(inplace=True)

# Convert columns to appropriate types
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')

# Remove rows where any conversion to numeric failed
df.dropna(subset=['quantity', 'sale_price', 'total_amount'], inplace=True)

# 1. Insights into Purchase Patterns

# Top Products by Sales
top_products = df.groupby('product_name')['quantity'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products by Sales:\n", top_products)

# Plot Top Products by Sales
plt.figure(figsize=(12, 8))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 10 Products by Sales Quantity')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product Name')
plt.show()

# Most Frequent Customers
top_customers = df['customer_name'].value_counts().head(10)
print("\nTop 10 Frequent Customers:\n", top_customers)

# Plot Top Customers by Frequency
plt.figure(figsize=(12, 8))
sns.barplot(x=top_customers.values, y=top_customers.index, palette='coolwarm')
plt.title('Top 10 Frequent Customers')
plt.xlabel('Number of Purchases')
plt.ylabel('Customer Name')
plt.show()

# Most Used Purchase Channels
channel_usage = df['purchase_channel'].value_counts()
print("\nPurchase Channel Usage:\n", channel_usage)

# Plot Channel Usage
plt.figure(figsize=(10, 6))
sns.barplot(x=channel_usage.values, y=channel_usage.index, palette='magma')
plt.title('Purchase Channel Usage')
plt.xlabel('Number of Purchases')
plt.ylabel('Channel')
plt.show()

# 2. Trend Analysis

# Extract Month and Day from Purchase Date
df['purchase_month'] = df['purchase_date'].dt.month
df['purchase_day'] = df['purchase_date'].dt.day

# Monthly Purchase Trends
monthly_trends = df['purchase_month'].value_counts().sort_index()
print("\nMonthly Purchase Trends:\n", monthly_trends)

# Plot Monthly Purchase Trends
plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_trends.index, y=monthly_trends.values, marker='o', color='green')
plt.title('Monthly Purchase Trends')
plt.xlabel('Month')
plt.ylabel('Number of Purchases')
plt.grid(True)
plt.show()

# Customer Loyalty Analysis
loyalty_analysis = df.groupby('customer_segment')['loyalty_points_earned'].mean()
print("\nAverage Loyalty Points Earned by Customer Segment:\n", loyalty_analysis)

# Plot Loyalty Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x=loyalty_analysis.index, y=loyalty_analysis.values, palette='viridis')
plt.title('Average Loyalty Points Earned by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Average Loyalty Points')
plt.show()

# 3. Attribute Relationships

# Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df[['quantity', 'sale_price', 'total_amount', 'profit_margin', 'loyalty_points_earned']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Different Attributes')
plt.show()

# Sentiment Impact on Average Sale Price
sentiment_sale_price = df.groupby('sentiment')['sale_price'].mean()
print("\nAverage Sale Price by Sentiment:\n", sentiment_sale_price)

# Plot Sentiment Impact on Sale Price
plt.figure(figsize=(10, 6))
sns.barplot(x=sentiment_sale_price.index, y=sentiment_sale_price.values, palette='RdYlGn')
plt.title('Average Sale Price by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Average Sale Price')
plt.show()
