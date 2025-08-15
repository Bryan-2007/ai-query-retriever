from flask import Flask, request, jsonify

app = Flask(__name__)

# Root URL for quick testing in browser
@app.route("/", methods=["GET"])
def home():
    return "✅ Webhook server is running", 200

# Webhook endpoint for POST requests
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json  # Get JSON payload
    print("📩 Webhook received:", data)
    
    # Respond to sender
    return jsonify({"status": "success", "message": "Webhook received"}), 200

if __name__ == "__main__":
    # Run locally on port 5000
    app.run(host="0.0.0.0", port=5000)
