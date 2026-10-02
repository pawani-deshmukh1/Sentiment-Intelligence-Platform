import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    print("1. Loading cleaned dataset...")
    data_path = os.path.join('..', 'data', 'Cleaned_Tweets.csv')
    df = pd.read_csv(data_path)
    
    df = df.dropna(subset=['cleaned_text'])
    X = df['cleaned_text']
    y = df['sentiment']

    print("2. Splitting data into Training (80%) and Testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Applying Balanced TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=8000,      # Reduced from 12000 to prevent overfitting
        min_df=3,               # Ignore words/phrases that appear less than 3 times total
        ngram_range=(1, 2),     # Keep bigrams (now they will capture "not good")
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("4. Training Logistic Regression Model...")
    # C=1.5 gives a perfect balance. It doesn't overfit like C=10.
    model = LogisticRegression(C=1.5, max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("5. Evaluating the Model...")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Performance ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("6. Saving the Upgraded Model and Vectorizer...")
    os.makedirs(os.path.join('..', 'models'), exist_ok=True)
    joblib.dump(vectorizer, os.path.join('..', 'models', 'tfidf_vectorizer.pkl'))
    joblib.dump(model, os.path.join('..', 'models', 'logistic_model.pkl'))
    print("Model files updated successfully!")

if __name__ == "__main__":
    train_model()