import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# --- One-Time NLTK Downloads ---
# NLTK needs to download its dictionary of stopwords and lemmatization rules.
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        # Load English stop words
        self.stop_words = set(stopwords.words('english'))
        
        # --- THE FIX: Remove negations from stop_words so we don't lose sentiment context ---
        negations = {'not', 'no', 'nor', 'none', 'cannot', 'isn', 'aren', 'wasn', 'weren', 
                     'hasn', 'haven', 'hadn', 'doesn', 'don', 'didn', 'won', 'wouldn', 
                     'shan', 'shouldn', 'mustn', 'can', 'couldn'}
        self.stop_words = self.stop_words - negations
        
        # Initialize Lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        """
        Takes a raw string, cleans it, and returns a processed string ready for ML.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Convert to lowercase so "Bad" and "bad" are treated as the same word
        text = text.lower()
        
        # 2. Remove URLs (e.g., http://link.com)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Remove user @mentions (e.g., @UnitedAirlines)
        text = re.sub(r'\@\w+|\#', '', text)
        
        # 4. Remove special characters, numbers, and punctuations (keep only alphabets)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 5. Tokenization (split sentence into individual words) & Remove Stopwords
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words
        ]
        
        # 6. Rejoin the words back into a single clean sentence
        return " ".join(cleaned_words)

# --- Code Execution Block ---
if __name__ == "__main__":
    print("Loading dataset...")
    # Navigate up one directory to access the data folder
    dataset_path = os.path.join('..', 'data', 'Tweets.csv')
    df = pd.read_csv(dataset_path)
    
    # We only need the text and the sentiment column
    df = df[['text', 'airline_sentiment']]
    df.columns = ['text', 'sentiment'] # Rename for easier access
    
    print("Cleaning text data... (this will take 10-20 seconds)")
    preprocessor = TextPreprocessor()
    
    # Apply the clean_text function to every row in the 'text' column
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    
    # Drop any rows that became empty after cleaning
    df = df[df['cleaned_text'].str.strip() != '']
    
    # Save the cleaned data to a new CSV so we don't have to clean it again during training
    cleaned_data_path = os.path.join('..', 'data', 'Cleaned_Tweets.csv')
    df.to_csv(cleaned_data_path, index=False)
    
    print("\n--- Preprocessing Complete ---")
    print("Example Before:", df['text'].iloc[0])
    print("Example After :", df['cleaned_text'].iloc[0])
    print(f"Cleaned dataset saved to: {cleaned_data_path}")