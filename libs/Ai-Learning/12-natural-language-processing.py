"""
12-natural-language-processing.py

Natural Language Processing (NLP)
---------------------------------
Learn NLP techniques for text analysis, sentiment analysis, and language
understanding. Explore both traditional and modern deep learning approaches.

What you'll learn
-----------------
1) Text preprocessing and tokenization
2) Feature extraction from text (TF-IDF, word embeddings)
3) Sentiment analysis and text classification
4) Language models and text generation
5) Modern NLP with transformers

Key Concepts
------------
- Text cleaning and normalization
- Bag-of-words and TF-IDF
- Word embeddings (Word2Vec, GloVe)
- Recurrent neural networks for text
- Transformer models and attention
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def text_preprocessing():
    """Demonstrate text preprocessing techniques"""
    print("=== Text Preprocessing Techniques ===")
    
    # Sample texts
    sample_texts = [
        "Hello! This is a SAMPLE text with UPPERCASE and lowercase words.",
        "I'm feeling great today!!! The weather is amazing.",
        "RT @user: Check out this amazing product! #awesome #product",
        "The quick brown fox jumps over the lazy dog. 123 numbers here!",
        "Text with   multiple    spaces and\nnewlines\n\nhere."
    ]
    
    print("Original texts:")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}")
    
    def preprocess_text(text, remove_stopwords=True, stem_words=True):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        
        # Stemming
        if stem_words:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    # Apply preprocessing
    processed_texts = []
    for text in sample_texts:
        processed = preprocess_text(text)
        processed_texts.append(processed)
    
    print("\nProcessed texts:")
    for i, text in enumerate(processed_texts):
        print(f"{i+1}. {text}")
    
    # Demonstrate different preprocessing levels
    preprocessing_levels = {
        'Original': sample_texts,
        'Lowercase only': [text.lower() for text in sample_texts],
        'Remove URLs/Mentions': [re.sub(r'http\S+|www\S+|https\S+|@\w+|#\w+', '', text) for text in sample_texts],
        'Remove punctuation': [re.sub(r'[^a-zA-Z\s]', '', text) for text in sample_texts],
        'Full preprocessing': processed_texts
    }
    
    # Visualize preprocessing effects
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (level, texts) in enumerate(preprocessing_levels.items()):
        if i < 6:  # Only show first 6 levels
            # Count word lengths
            word_lengths = [len(text.split()) for text in texts]
            
            axes[i].hist(word_lengths, bins=10, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{level}\nAvg words: {np.mean(word_lengths):.1f}')
            axes[i].set_xlabel('Number of words')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('text_preprocessing.png', dpi=300, bbox_inches='tight')
    print("Text preprocessing visualization saved as 'text_preprocessing.png'")
    plt.show()
    
    return processed_texts

def text_feature_extraction():
    """Demonstrate text feature extraction techniques"""
    print("\n=== Text Feature Extraction ===")
    
    # Sample documents
    documents = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "Cats and dogs are pets",
        "I love programming in Python",
        "Machine learning is fascinating",
        "Python is great for data science",
        "Dogs are loyal animals",
        "Cats are independent creatures"
    ]
    
    print("Sample documents:")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc}")
    
    # 1. Bag of Words
    vectorizer_bow = CountVectorizer()
    bow_matrix = vectorizer_bow.fit_transform(documents)
    bow_features = bow_matrix.toarray()
    
    print(f"\nBag of Words features shape: {bow_features.shape}")
    print(f"Vocabulary: {vectorizer_bow.get_feature_names_out()}")
    
    # 2. TF-IDF
    vectorizer_tfidf = TfidfVectorizer()
    tfidf_matrix = vectorizer_tfidf.fit_transform(documents)
    tfidf_features = tfidf_matrix.toarray()
    
    print(f"\nTF-IDF features shape: {tfidf_features.shape}")
    
    # Visualize feature matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bag of Words heatmap
    sns.heatmap(bow_features, annot=True, fmt='d', cmap='Blues', 
                xticklabels=vectorizer_bow.get_feature_names_out(),
                yticklabels=[f'Doc {i+1}' for i in range(len(documents))],
                ax=axes[0])
    axes[0].set_title('Bag of Words Features')
    axes[0].set_xlabel('Words')
    axes[0].set_ylabel('Documents')
    
    # TF-IDF heatmap
    sns.heatmap(tfidf_features, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=vectorizer_tfidf.get_feature_names_out(),
                yticklabels=[f'Doc {i+1}' for i in range(len(documents))],
                ax=axes[1])
    axes[1].set_title('TF-IDF Features')
    axes[1].set_xlabel('Words')
    axes[1].set_ylabel('Documents')
    
    plt.tight_layout()
    plt.savefig('text_features.png', dpi=300, bbox_inches='tight')
    print("Text features visualization saved as 'text_features.png'")
    plt.show()
    
    # Document similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(tfidf_features)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=[f'Doc {i+1}' for i in range(len(documents))],
                yticklabels=[f'Doc {i+1}' for i in range(len(documents))])
    plt.title('Document Similarity Matrix (TF-IDF)')
    plt.tight_layout()
    plt.savefig('document_similarity.png', dpi=300, bbox_inches='tight')
    print("Document similarity visualization saved as 'document_similarity.png'")
    plt.show()
    
    return bow_features, tfidf_features, vectorizer_bow, vectorizer_tfidf

def text_classification():
    """Demonstrate text classification with different algorithms"""
    print("\n=== Text Classification ===")
    
    # Load 20 newsgroups dataset
    try:
        newsgroups = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'sci.space', 'comp.graphics'])
        texts = newsgroups.data
        labels = newsgroups.target
        target_names = newsgroups.target_names
    except:
        # Fallback to synthetic data
        texts = [
            "I love programming and coding in Python",
            "Space exploration is fascinating and important",
            "Graphics and design are creative fields",
            "Python is great for machine learning",
            "The universe is vast and mysterious",
            "Computer graphics involve rendering and animation",
            "Data science uses Python extensively",
            "Astronomy studies celestial objects",
            "Software development requires programming skills",
            "Cosmic phenomena are incredible to study"
        ]
        labels = [0, 1, 2, 0, 1, 2, 0, 1, 0, 1]  # 0: programming, 1: space, 2: graphics
        target_names = ['Programming', 'Space', 'Graphics']
    
    print(f"Dataset: {len(texts)} documents, {len(target_names)} classes")
    print(f"Classes: {target_names}")
    
    # Preprocess texts
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42
    )
    
    # Extract features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    # Train different classifiers
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("\nClassification Results:")
    print("-" * 50)
    
    for name, classifier in classifiers.items():
        # Train
        classifier.fit(X_train_features, y_train)
        
        # Predict
        y_pred = classifier.predict(X_test_features)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classifier': classifier
        }
        
        print(f"{name:20}: {accuracy:.3f}")
    
    # Detailed classification report for best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_predictions = results[best_model_name]['predictions']
    
    print(f"\nDetailed Classification Report ({best_model_name}):")
    print(classification_report(y_test, best_predictions, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('text_classification.png', dpi=300, bbox_inches='tight')
    print("Text classification visualization saved as 'text_classification.png'")
    plt.show()
    
    return results, vectorizer

def sentiment_analysis():
    """Demonstrate sentiment analysis"""
    print("\n=== Sentiment Analysis ===")
    
    # Sample texts with known sentiment
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Terrible service.",
        "The movie was okay, nothing special but not bad either.",
        "Fantastic! Highly recommend this to everyone.",
        "Disappointing quality, expected much better.",
        "Great value for money, very satisfied with purchase.",
        "Awful customer support, will never buy again.",
        "Excellent quality and fast delivery, very happy!",
        "Average product, does what it's supposed to do.",
        "Outstanding performance, exceeded my expectations."
    ]
    
    # Manual sentiment labels (0: negative, 1: neutral, 2: positive)
    true_labels = [2, 0, 1, 2, 0, 2, 0, 2, 1, 2]
    sentiment_names = ['Negative', 'Neutral', 'Positive']
    
    print("Sample texts and true sentiments:")
    for i, (text, label) in enumerate(zip(texts, true_labels)):
        print(f"{i+1}. [{sentiment_names[label]}] {text}")
    
    # Preprocess texts
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()
    
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Extract features
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    features = vectorizer.fit_transform(processed_texts)
    
    # Train sentiment classifier
    classifier = LogisticRegression(random_state=42)
    classifier.fit(features, true_labels)
    
    # Make predictions
    predictions = classifier.predict(features)
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\nSentiment Analysis Accuracy: {accuracy:.3f}")
    
    # Show predictions
    print("\nPredictions:")
    for i, (text, true_label, pred_label) in enumerate(zip(texts, true_labels, predictions)):
        status = "✓" if true_label == pred_label else "✗"
        print(f"{status} [{sentiment_names[pred_label]}] {text[:50]}...")
    
    # Feature importance for sentiment
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_
    
    # Get most important features for each class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, sentiment in enumerate(sentiment_names):
        # Get top 10 features for this sentiment
        top_indices = np.argsort(coefficients[i])[-10:]
        top_features = [feature_names[idx] for idx in top_indices]
        top_weights = coefficients[i][top_indices]
        
        axes[i].barh(range(len(top_features)), top_weights)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features)
        axes[i].set_title(f'Top Features - {sentiment}')
        axes[i].set_xlabel('Weight')
    
    plt.tight_layout()
    plt.savefig('sentiment_features.png', dpi=300, bbox_inches='tight')
    print("Sentiment features visualization saved as 'sentiment_features.png'")
    plt.show()
    
    return classifier, vectorizer, accuracy

def neural_language_model():
    """Build a simple neural language model"""
    print("\n=== Neural Language Model ===")
    
    # Sample text data
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "Cats and dogs are pets",
        "I love programming in Python",
        "Machine learning is fascinating",
        "Python is great for data science",
        "Dogs are loyal animals",
        "Cats are independent creatures",
        "Programming requires practice and patience",
        "Data science combines statistics and programming"
    ]
    
    # Preprocess texts
    processed_texts = [text.lower() for text in texts]
    
    # Tokenize texts
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(processed_texts)
    
    # Pad sequences
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Max sequence length: {max_length}")
    print(f"Sample sequence: {padded_sequences[0]}")
    
    # Create training data (predict next word)
    X = []
    y = []
    
    for sequence in padded_sequences:
        for i in range(1, len(sequence)):
            if sequence[i] != 0:  # Skip padding
                X.append(sequence[:i])
                y.append(sequence[i])
    
    # Pad input sequences
    X = pad_sequences(X, maxlen=max_length-1, padding='pre')
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Build language model
    vocab_size = len(tokenizer.word_index) + 1
    
    model = models.Sequential([
        layers.Embedding(vocab_size, 50, input_length=max_length-1),
        layers.LSTM(100, return_sequences=True),
        layers.LSTM(100),
        layers.Dense(100, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Language Model Architecture:")
    model.summary()
    
    # Train model
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Generate text
    def generate_text(model, tokenizer, seed_text, max_length=5):
        for _ in range(max_length):
            # Tokenize seed text
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
            
            # Predict next word
            predicted = model.predict(token_list, verbose=0)
            predicted_id = np.argmax(predicted, axis=1)[0]
            
            # Convert back to word
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_id:
                    output_word = word
                    break
            
            seed_text += " " + output_word
        
        return seed_text
    
    # Generate some text
    print("\nGenerated text samples:")
    seed_texts = ["the cat", "python is", "dogs are"]
    
    for seed in seed_texts:
        generated = generate_text(model, tokenizer, seed, max_length=3)
        print(f"'{seed}' -> '{generated}'")
    
    # Visualize training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Language Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('language_model_training.png', dpi=300, bbox_inches='tight')
    print("Language model training visualization saved as 'language_model_training.png'")
    plt.show()
    
    return model, tokenizer, history

def word_embeddings_demo():
    """Demonstrate word embeddings concepts"""
    print("\n=== Word Embeddings Demo ===")
    
    # Sample texts for word embeddings
    texts = [
        "king queen man woman",
        "cat dog animal pet",
        "python programming coding",
        "machine learning artificial intelligence",
        "data science analytics",
        "neural network deep learning",
        "computer science technology",
        "mathematics statistics probability"
    ]
    
    # Preprocess and tokenize
    processed_texts = [text.lower().split() for text in texts]
    
    # Create vocabulary
    all_words = []
    for text in processed_texts:
        all_words.extend(text)
    
    vocabulary = list(set(all_words))
    word_to_id = {word: i for i, word in enumerate(vocabulary)}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Vocabulary: {vocabulary}")
    
    # Create co-occurrence matrix (simplified)
    window_size = 2
    cooccurrence_matrix = np.zeros((len(vocabulary), len(vocabulary)))
    
    for text in processed_texts:
        for i, word in enumerate(text):
            if word in word_to_id:
                word_id = word_to_id[word]
                # Look at surrounding words
                for j in range(max(0, i - window_size), min(len(text), i + window_size + 1)):
                    if i != j and text[j] in word_to_id:
                        context_id = word_to_id[text[j]]
                        cooccurrence_matrix[word_id][context_id] += 1
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(cooccurrence_matrix)
    
    # Visualize word embeddings
    plt.figure(figsize=(12, 8))
    
    # Plot words
    for i, word in enumerate(vocabulary):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100, alpha=0.7)
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Word Embeddings (2D PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.savefig('word_embeddings.png', dpi=300, bbox_inches='tight')
    print("Word embeddings visualization saved as 'word_embeddings.png'")
    plt.show()
    
    # Show similar words (based on cosine similarity)
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity(cooccurrence_matrix)
    
    # Find most similar words for some examples
    example_words = ['king', 'python', 'machine']
    
    print("\nMost similar words:")
    for word in example_words:
        if word in word_to_id:
            word_id = word_to_id[word]
            similar_indices = np.argsort(similarities[word_id])[-4:-1][::-1]  # Top 3 (excluding self)
            
            print(f"\nWords similar to '{word}':")
            for idx in similar_indices:
                similar_word = id_to_word[idx]
                similarity = similarities[word_id][idx]
                print(f"  {similar_word}: {similarity:.3f}")
    
    return cooccurrence_matrix, embeddings_2d, word_to_id

def main():
    """Main function to run all NLP demonstrations"""
    print("📝 Natural Language Processing (NLP)")
    print("=" * 50)
    
    # Text preprocessing
    text_preprocessing()
    
    # Text feature extraction
    text_feature_extraction()
    
    # Text classification
    text_classification()
    
    # Sentiment analysis
    sentiment_analysis()
    
    # Neural language model
    neural_language_model()
    
    # Word embeddings
    word_embeddings_demo()
    
    print("\n" + "=" * 50)
    print("Lesson 12 Complete!")
    print("Congratulations! You've completed the AI Learning Course!")
    print("Key takeaway: NLP enables machines to understand, interpret, and generate human language")
    print("\nNext Steps:")
    print("- Practice with real-world datasets")
    print("- Explore advanced NLP models (BERT, GPT)")
    print("- Build end-to-end NLP applications")
    print("- Stay updated with latest AI research")

if __name__ == "__main__":
    main()
