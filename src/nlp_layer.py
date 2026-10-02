import numpy as np

class NLPIntelligence:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        # Get all the words the model knows
        self.feature_names = vectorizer.get_feature_names_out()
        
        # --- Rule-Based Dictionaries for Aspects ---
        self.aspect_keywords = {
            "Customer Service": ["staff", "agent", "service", "rude", "helpful", "call"],
            "Baggage & Belongings": ["bag", "luggage", "suitcase", "lost"],
            "Flight Experience": ["flight", "delayed", "seat", "plane", "hours", "wait"],
            "Booking & UI": ["app", "website", "booking", "ticket", "cancel"]
        }

    def extract_aspects_and_emotions(self, text, sentiment):
        """
        Uses rule-based logic to find aspects and determine emotion/suggestions.
        """
        text_lower = text.lower()
        found_aspects = []
        primary_emotion = {"label": "Neutral", "confidence": 1.0}
        suggestions = []

        # 1. Aspect Extraction
        for aspect, keywords in self.aspect_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                # If the aspect is found, we assume its sentiment matches the overall sentence
                found_aspects.append({
                    "aspect_name": aspect,
                    "sentiment": sentiment
                })

        # 2. Emotion & Suggestion Routing
        if sentiment == "negative":
            if any(a['aspect_name'] == "Baggage & Belongings" for a in found_aspects):
                primary_emotion = {"label": "Anxiety / Frustration", "confidence": 0.88}
                suggestions.append("Action required: Please contact the baggage claim desk with your PNR number immediately.")
            elif any(a['aspect_name'] == "Flight Experience" for a in found_aspects):
                primary_emotion = {"label": "Exhaustion / Frustration", "confidence": 0.85}
                suggestions.append("We apologize for the delay. You may be entitled to lounge access or food vouchers.")
            elif any(a['aspect_name'] == "Customer Service" for a in found_aspects):
                primary_emotion = {"label": "Anger", "confidence": 0.90}
                suggestions.append("We take staff behavior seriously. Please consider raising a formal grievance ticket.")
            else:
                primary_emotion = {"label": "Disappointment", "confidence": 0.75}
                suggestions.append("We are sorry to hear about your experience. Please reach out to support.")
                
        elif sentiment == "positive":
            primary_emotion = {"label": "Joy / Satisfaction", "confidence": 0.92}
            suggestions.append("Thank you for your feedback! We hope to serve you again soon.")

        return found_aspects, primary_emotion, suggestions

    def get_explainability(self, cleaned_text, predicted_class_name):
        """
        Extracts the exact words that caused the model to predict the sentiment.
        """
        vec = self.vectorizer.transform([cleaned_text])
        non_zero_indices = vec.nonzero()[1]
        
        if len(non_zero_indices) == 0:
            return {"contributing_words": [], "summary": "No strong sentiment words identified."}

        # Find the mathematical index for the predicted class
        class_index = list(self.model.classes_).index(predicted_class_name)
        
        word_weights = []
        for index in non_zero_indices:
            word = self.feature_names[index]
            # Get the weight of this word FOR THE PREDICTED CLASS
            weight = self.model.coef_[class_index][index]
            word_weights.append((word, weight))
            
        # Sort words by their mathematical weight (highest weight first)
        # The words at the top are the ones that pushed the model to make this prediction
        word_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Grab the top 5 words that have a meaningful impact (weight > 0.2)
        top_contributors = [word for word, weight in word_weights if weight > 0.2][:5]

        return {
            "contributing_words": top_contributors,
            "summary": f"The model detected strong {predicted_class_name} sentiment driven by words like: {', '.join(top_contributors)}"
        }