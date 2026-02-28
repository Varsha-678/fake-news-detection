#!/usr/bin/env python3
"""
Fake News Detection System
AI-Powered tool to detect fake news articles using Machine Learning
Developed for AICTE Internship Capstone Project
Author: Varsha K, Sri Ramakrishna Institute of Technology
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    """
    A comprehensive Fake News Detection System using Multiple ML Models
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.model_name = None
        
    def preprocess_text(self, text):
        """Clean and preprocess the text data"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_data(self, filepath=None):
        """Load the fake news dataset"""
        try:
            if filepath:
                df = pd.read_csv(filepath)
            else:
                print("Using sample dataset.")
                df = self._create_sample_data()
            print(f"Dataset loaded successfully! Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        fake_texts = [
            "Scientists discover miracle cure for all diseases using common household items",
            "Breaking: Aliens land in Times Square, New York City",
            "Local man discovers fountain of youth in his backyard",
            "Government announces free money for all citizens starting next month",
            "Study shows chocolate cures cancer, big pharma hiding the truth"
        ] * 100
        
        real_texts = [
            "The stock market showed gains today as technology stocks led the rally",
            "Scientists publish new research on climate change impacts",
            "Local community comes together to support flood victims",
            "New education policy announced by government officials",
            "Sports team wins championship after thrilling final match"
        ] * 100
        
        data = {
            'text': fake_texts + real_texts,
            'label': [0] * len(fake_texts) + [1] * len(real_texts)
        }
        return pd.DataFrame(data)
    
    def train(self, df, text_column='text', label_column='label'):
        """Train all models and select the best one"""
        print("\nTraining Fake News Detection Models...")
        df['clean_text'] = df[text_column].apply(self.preprocess_text)
        X = df['clean_text']
        y = df[label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        best_accuracy = 0
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model
                self.model_name = name
        
        print(f"\nBest Model: {self.model_name}")
        print(f"Best Accuracy: {best_accuracy*100:.2f}%")
        return best_accuracy
    
    def predict(self, text):
        """Predict if a news article is fake or real"""
        if self.best_model is None:
            return "Please train the model first!"
        clean_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([clean_text])
        prediction = self.best_model.predict(text_vec)[0]
        probability = self.best_model.predict_proba(text_vec)[0]
        return {
            'prediction': 'REAL' if prediction == 1 else 'FAKE',
            'confidence': max(probability) * 100,
            'fake_probability': probability[0] * 100,
            'real_probability': probability[1] * 100
        }
    
    def save_model(self, path='fake_news_model.pkl'):
        """Save the trained model"""
        model_data = {'model': self.best_model, 'model_name': self.model_name, 'vectorizer': self.vectorizer}
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")


def main():
    print("="*60)
    print("FAKE NEWS DETECTION SYSTEM")
    print("="*60)
    
    detector = FakeNewsDetector()
    df = detector.load_data()
    detector.train(df)
    detector.save_model()
    
    print("\nTesting predictions:")
    test_articles = [
        "Breaking: Scientists confirm that drinking coffee daily can extend your life",
        "The local government announced new infrastructure projects",
        "Miracle pill discovered that allows humans to breathe underwater"
    ]
    
    for article in test_articles:
        result = detector.predict(article)
        print(f"\nArticle: {article[:60]}...")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f}%)")


if __name__ == "__main__":
    main()
