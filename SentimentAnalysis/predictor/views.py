from django.shortcuts import render

# Create your views here.
import joblib
import json
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.feature_extraction.text import CountVectorizer
from django.views.decorators.csrf import csrf_exempt

# Load model and important words
model = joblib.load('predictor/logistic_regression_model.pkl')
with open('predictor/important_words.json') as f:
    important_words = json.load(f)

# Create a function to convert text to features
def text_to_features(text):
    vectorizer = CountVectorizer(vocabulary=important_words)
    text_features = vectorizer.transform([text]).toarray()
    return text_features

def index(request):
    return render(request, 'predictor/index.html')

@csrf_exempt
def predict_sentiment(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        review = data.get('review', '')
        features = text_to_features(review)
        prediction = model.predict(features)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        return JsonResponse({'sentiment': sentiment})
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)
