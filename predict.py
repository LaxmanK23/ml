# predict.py
import joblib, argparse

LABELS = {0: "ham", 1: "spam"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sms_spam_model.pkl",
                        help="Path to trained model file")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Decision threshold for spam (default=0.50)")
    args = parser.parse_args()

    model = joblib.load(args.model)

    print(f"Spam detector ready (threshold={args.threshold}). Type a message (or 'quit').")
    while True:
        msg = input("> ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        prob_spam = float(model.predict_proba([msg])[0][1])
        pred = int(prob_spam >= args.threshold)
        print(f"{LABELS[pred]}  (spam probability: {prob_spam:.2f})")
