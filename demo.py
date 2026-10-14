import joblib
from preprocess import clean_text

pipe = joblib.load("models/sms_spam_pipeline.pkl")
print("=" * 60)
print("      SMS SPAM DETECTOR - LIVE DEMO")
print("=" * 60)

test_messages = [
    "Congratulations! You won a free iPhone. Click here now.",
    "Hi, are we meeting tomorrow at 3 PM?",
    "URGENT! Your mobile number has won 2000 pounds. Call now.",
    "Hey mom, I'll be home by 7 PM.",
]

for msg in test_messages:
    cleaned = clean_text(msg)
    prob = pipe.predict_proba([cleaned])[0][1]
    label = "SPAM" if prob >= 0.5 else "HAM"
    print(f"\nMessage  : {msg}")
    print(f"Result   : {label}")
    print(f"Confidence: {prob:.2%}")

print("\n" + "=" * 60)
print("DEMO COMPLETED SUCCESSFULLY!")
print("=" * 60)