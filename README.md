# 🧠 Multimodal Sentiment & Aspect Intelligence Platform

**Domain:** Data Retrieval and Text Mining (DRTM)  
**Status:** Core ML Pipeline & NLP Intelligence Engine — **Active / Completed**

This repository contains the core machine learning and natural language processing backend for a multimodal sentiment analysis platform. Unlike standard binary classifiers, this system incorporates Aspect-Based Sentiment Analysis (ABSA), Emotion Detection, and Explainable AI (XAI) to provide transparent, context-aware insights from raw text.

---

## ✨ Core Features (Phases 1 & 2)

*   **Explainable ML Core:** Utilizes TF-IDF vectorization and a mathematically robust Logistic Regression model (78.6% baseline accuracy) optimized with N-grams and Sublinear TF.
*   **Aspect Mining:** Extracts specific entities (e.g., *Customer Service*, *Baggage*, *Flight Experience*) from unstructured text.
*   **Emotion Detection:** Maps sentiment and aspects to primary human emotions (e.g., *Frustration*, *Joy*, *Anxiety*).
*   **Explainability (XAI):** Extracts exact contextual words that contributed to the model's prediction by analyzing coefficient weights.
*   **Context-Aware Suggestions:** Generates actionable, deterministic suggestions based on identified aspects and emotions.

---

## 📂 Repository Structure

```text
sentiment-intelligence-platform/
│
├── data/                  # Contains the raw and preprocessed datasets
├── models/                # Serialized ML artifacts (.pkl files)
├── src/                   
│   ├── preprocess.py      # Text cleaning, tokenization & lemmatization pipeline
│   ├── train.py           # Model training, hyperparameter tuning & evaluation
│   ├── nlp_layer.py       # Rule-based Aspect, Emotion & Explainability logic
│   └── predict.py         # Full pipeline integration and inference engine
│
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

---
</> Markdown
## 🛠️ Local Setup & Installation
To run the machine learning engine locally, follow these steps:

1. Create and activate a virtual environment:
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

2. Install project dependencies:
pip install -r requirements.txt

3. Execute the CLI Inference Engine:
cd src
python predict.py
(Enter a sample review into the terminal to generate a full sentiment and aspect analysis.)

---

## 📜 API Response Schema
The system utilizes a Contract-First Architecture to ensure seamless frontend and backend decoupling. All HTTP POST requests to the future analysis endpoint will return the following strictly typed JSON structure:

JSON
{
  "meta": {
    "input_modality": "text", 
    "processing_time_ms": 42
  },
  "core_analysis": {
    "text": "I waited for 3 hours and lost my luggage, totally terrible experience.",
    "overall_sentiment": "negative",
    "confidence_score": 0.99
  },
  "nlp_intelligence": {
    "primary_emotion": {
      "label": "Anxiety / Frustration",
      "confidence": 0.88
    },
    "aspects": [
      {
        "aspect_name": "Baggage & Belongings",
        "sentiment": "negative"
      },
      {
        "aspect_name": "Flight Experience",
        "sentiment": "negative"
      }
    ],
    "explainability": {
      "contributing_words": ["hour", "luggage", "terrible", "lost", "experience"],
      "summary": "The model detected strong negative sentiment driven by words like: hour, luggage, terrible, lost, experience"
    },
    "suggestions": [
      "Action required: Please contact the baggage claim desk with your PNR number immediately."
    ]
  }
}

---

## 🚀 Development Roadmap
Phase 1 [Completed]: Core TF-IDF + Logistic Regression Model Initialization.

Phase 2 [Completed]: NLP Layer Integration (Aspects, Emotions, Explainability).

Phase 3 [Pending]: FastAPI Backend Wrapper & Voice Input Integration.

Phase 4 [Pending]: React-based Analytics Dashboard & UI Development.
