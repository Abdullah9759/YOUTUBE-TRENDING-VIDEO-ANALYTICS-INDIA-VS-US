import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load India and US datasets
in_df = pd.read_csv("INvideos.csv")
us_df = pd.read_csv("USvideos.csv")

# Tag country
in_df["Country"] = "India"
us_df["Country"] = "US"

# Combine datasets
df = pd.concat([in_df, us_df], ignore_index=True)

# Clean data
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert publish_time and extract hour and day
df["publish_time"] = pd.to_datetime(df["publish_time"], errors='coerce')
df["publish_hour"] = df["publish_time"].dt.hour
df["publish_day"] = df["publish_time"].dt.day_name()

# Sentiment Analysis on title
analyzer = SentimentIntensityAnalyzer()
df["title_sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment_label"] = df["title_sentiment"].apply(
    lambda x: "Positive" if x > 0.2 else ("Negative" if x < -0.2 else "Neutral")
)

# Export final dataset
df.to_csv("C:/Users/iqram/Downloads/archive (9)/YouTube_Final.csv", index=False)
print(" YouTube_Final.csv created successfully in 3_cleaned_data folder.")
