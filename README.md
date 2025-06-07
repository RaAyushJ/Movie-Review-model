# Sentiment Analysis on Movie Reviews

This project performs sentiment analysis on the IMDB dataset using a Logistic Regression classifier. The dataset contains 50,000 movie reviews labeled as either **positive** or **negative**.

## Objective

- Load and explore the IMDB dataset of movie reviews and sentiment labels.
- Preprocess text data using TF-IDF vectorization.
- Split the dataset into training and testing sets.
- Train a Logistic Regression classifier to predict sentiment.
- Evaluate the model using classification metrics such as precision, recall, and F1-score.

## Dataset

- Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Format: CSV with 50,000 rows and 2 columns:
  - `review`: the movie review text
  - `sentiment`: the label (`positive` or `negative`)

## Tools Used

- Python (Google Colab)
- scikit-learn
- pandas
- TfidfVectorizer

## Model Training

The text reviews were converted into numerical features using TF-IDF vectorization. The dataset was split into 80% training and 20% testing data. A Logistic Regression model was trained on the vectorized data.

## Evaluation Results

**Accuracy:** 0.894  
**Precision:** 0.88  
**Recall:** 0.91  
**F1 Score:** 0.896  

### Classification Report:

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Negative  | 0.91      | 0.88   | 0.89     | 4961    |
| Positive  | 0.88      | 0.91   | 0.90     | 5039    |
| **Accuracy** |         |        | **0.89** | 10000   |
| Macro Avg | 0.89      | 0.89   | 0.89     | 10000   |
| Weighted Avg | 0.89   | 0.89   | 0.89     | 10000   |

## Output

The trained model (`logistic_model.pkl`) and the vectorizer (`tfidf_vectorizer.pkl`) were saved for future use and can be used to predict the sentiment of new movie reviews.

## How to Use

1. Load the saved `logistic_model.pkl` and `tfidf_vectorizer.pkl`.
2. Transform new text using the TF-IDF vectorizer.
3. Use the model to predict sentiment.

---

**Example:**

```python
import pickle

# Load model and vectorizer
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Predict sentiment
review = ["The movie was absolutely wonderful and the acting was superb."]
vectorized = tfidf.transform(review)
prediction = model.predict(vectorized)

print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
