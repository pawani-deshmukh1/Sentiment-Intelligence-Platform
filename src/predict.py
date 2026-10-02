import os
import joblib
import json
from preprocess import TextPreprocessor
from nlp_layer import NLPIntelligence

class FullSentimentPipeline:
    def __init__(self):
        print("Loading ML models and NLP Engine...")
        model_path = os.path.join('..', 'models', 'logistic_model.pkl')
        vec_path = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vec_path)
        
        self.preprocessor = TextPreprocessor()
        # Initialize Khushalika's NLP Layer
        self.nlp_engine = NLPIntelligence(self.model, self.vectorizer)

    def analyze(self, text):
        # --- PHASE 1: Pawani's ML Core ---
        cleaned_text = self.preprocessor.clean_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        
        prediction = self.model.predict(vectorized_text)[0]
        confidence = max(self.model.predict_proba(vectorized_text)[0])
        
        if confidence < 0.60:
            prediction = 'neutral'

        # --- PHASE 2: Khushalika's NLP Layer ---
        aspects, emotion, suggestions = self.nlp_engine.extract_aspects_and_emotions(text, prediction)
        explainability = self.nlp_engine.get_explainability(cleaned_text, prediction)

        # --- Assemble the Final API Contract ---
        final_json = {
            "core_analysis": {
                "text": text,
                "overall_sentiment": prediction,
                "confidence_score": round(confidence, 2)
            },
            "nlp_intelligence": {
                "primary_emotion": emotion,
                "aspects": aspects,
                "explainability": explainability,
                "suggestions": suggestions
            }
        }
        return final_json

if __name__ == "__main__":
    pipeline = FullSentimentPipeline()
    print("\n--- Sentiment & Aspect Intelligence Platform ---")
    
    while True:
        user_input = input("\nEnter text (or 'exit'): ")
        if user_input.lower() == 'exit':
            break
            
        result = pipeline.analyze(user_input)
        
        # Print the output in beautiful JSON format
        print("\n" + json.dumps(result, indent=2))