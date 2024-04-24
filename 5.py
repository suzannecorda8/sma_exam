import pandas as pd
import matplotlib.pyplot as plt


# Read the Apple and Samsung datasets
apple_df = pd.read_csv('apple.csv')
samsung_df = pd.read_csv('samsung.csv')


# Function to categorize sentiment label
def categorize_sentiment_label(sentiment_label):
    if sentiment_label == 'positive':
        return 'Positive'
    elif sentiment_label == 'negative':
        return 'Negative'
    else:
        return 'Neutral'


# Categorize sentiment label for Apple dataset
apple_df['Sentiment'] = apple_df['sentiment_label'].apply(categorize_sentiment_label)


# Categorize sentiment label for Samsung dataset
samsung_df['Sentiment'] = samsung_df['sentiment_label'].apply(categorize_sentiment_label)


# Combine the datasets
combined_df = pd.concat([apple_df, samsung_df])


# Function to calculate sentiment counts
def calculate_sentiment_counts(df):
    sentiment_counts = df['Sentiment'].value_counts()
    counts = {
        'Positive': sentiment_counts['Positive'] if 'Positive' in sentiment_counts else 0,
        'Neutral': sentiment_counts['Neutral'] if 'Neutral' in sentiment_counts else 0,
        'Negative': sentiment_counts['Negative'] if 'Negative' in sentiment_counts else 0
    }
    return counts


# Calculate sentiment counts for Apple and Samsung
apple_sentiment_counts = calculate_sentiment_counts(apple_df)
samsung_sentiment_counts = calculate_sentiment_counts(samsung_df)




print(apple_sentiment_counts)
# Plotting for Apple
plt.figure(figsize=(5, 5))
plt.bar(apple_sentiment_counts.keys(), apple_sentiment_counts.values(), color=['green', 'blue', 'red'])
plt.title('Sentiment Analysis for Apple Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


print(samsung_sentiment_counts)
# Plotting for Samsung
plt.figure(figsize=(5, 5))
plt.bar(samsung_sentiment_counts.keys(), samsung_sentiment_counts.values(), color=['green', 'blue', 'red'])
plt.title('Sentiment Analysis for Samsung Dataset')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
# Create a DataFrame for combined sentiment counts
combined_sentiment_counts = pd.DataFrame({
    'Apple': apple_sentiment_counts,
    'Samsung': samsung_sentiment_counts
})


# Plotting
combined_sentiment_counts.plot(kind='bar', figsize=(5, 5))
plt.title('Comparative Sentiment Counts for Apple and Samsung Reviews')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Brand')
plt.tight_layout()
plt.show()
