import os
import sys

sys.path.append(os.path.abspath("."))

from src.app.predictor import predict_sentiment

print("=" * 50)
print("HINGLISH SENTIMENT ANALYZER")
print("=" * 50)

while True:

    text = input("\nEnter text (type 'exit' to quit): ")

    if text.lower() == "exit":
        break

    result = predict_sentiment(text)

    print("\nPrediction")
    print(result["label"])

    print("\nConfidence")
    print(f"{result['confidence']*100:.2f}%")

    print("\nProbabilities")

    for label, prob in result["probabilities"].items():

        print(f"{label:<10}: {prob*100:.2f}%")