Below is a clear step-by-step guide to help you obtain a `GOOGLE_API_KEY` (for Gemini / Google AI APIs) and verify that it works correctly.

---

# 🔑 How to Get Your `GOOGLE_API_KEY`

Google provides API access through **Google AI Studio** and the Google Cloud platform.

---

## Step 1 — Go to Google AI Studio

Open:

👉 [https://aistudio.google.com/](https://aistudio.google.com/)

Sign in with your Google account.

---

## Step 2 — Create an API Key

1. Click **“Get API key”** (usually top right or in settings).
2. Select or create a Google Cloud project.
3. Click **“Create API key”**.
4. Copy the generated key.

It will look like:

```
AIzaSyXXXXXXXXXXXXXXX
```

Keep it private. Do not commit it to GitHub.

---

## 💳 Billing & Free Tier

* Google provides **free usage quota** for Gemini APIs.
* However, you may need to:

  * Enable billing on your Google Cloud project.
  * Add a payment method.
* You will only be charged if you exceed the free quota.

You can check usage and quota in the Google Cloud Console:
👉 [https://console.cloud.google.com/](https://console.cloud.google.com/)

---

# 🖥️ Set the Environment Variable

On macOS / Linux:

```bash
export GOOGLE_API_KEY="AIzaSyXXXXXXXXXXXXXXX"
```

On Windows (PowerShell):

```powershell
setx GOOGLE_API_KEY "AIzaSyXXXXXXXXXXXXXXX"
```

Verify:

```bash
echo $GOOGLE_API_KEY
```

---

# ✅ Verify Your API Key (Minimal Python Test)

## Step 1 — Install the Google Generative AI SDK

```bash
pip install google-generativeai
```

---

## Step 2 — Create a Test Script

Create `test_google.py`:

```python
import os
from google import genai

api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY is not set.")
    exit(1)

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents="Say hello."
    )

    print("✅ API key is valid!")
    print("Model response:", response.text)

except Exception as e:
    print("❌ API key is invalid or not working.")
    print("Error:", e)
```

Run it:

```bash
python test_google.py
```

---

## Expected Output (If Valid)

```
✅ API key is valid!
Model response: Hello!
```

---

# 🔍 If It Fails

Common issues:

* ❌ `GOOGLE_API_KEY` not exported
* ❌ Billing not enabled
* ❌ API not enabled in project
* ❌ Quota exceeded
* ❌ Network/firewall restrictions

---

# 🔐 Security Tips

* Never hard-code your API key.
* Add `.env` files to `.gitignore`.
* Rotate keys if exposed.
* Use separate keys for development and production.