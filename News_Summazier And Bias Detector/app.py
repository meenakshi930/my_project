import os
import json
import requests
import re
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

app = Flask(__name__, template_folder="templates")


API_KEY = os.environ.get("AIzaSyDxKpQtqXpkZ9ok6CisMSb_EvKLvzK-qzk")

genai.configure(api_key=API_KEY)

MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(MODEL_NAME)

def extract_text_from_url(url, max_chars=40000):
    """Extracts main text content from a news article URL."""
    headers = {"User-Agent": "NewsBiasDetector/1.0"}
    resp = requests.get(url, headers=headers, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    
    article_tag = soup.find("article")
    if article_tag:
        text = " ".join(p.get_text(" ", strip=True) for p in article_tag.find_all("p"))
        if len(text) > 200:
            return text[:max_chars]

    
    main_tag = soup.find("main")
    if main_tag:
        text = " ".join(p.get_text(" ", strip=True) for p in main_tag.find_all("p"))
        if len(text) > 200:
            return text[:max_chars]

    
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(paragraphs)
    if len(text) < 100:
        raise RuntimeError("Extracted content too short.")
    return text[:max_chars]

def build_prompt(article_text: str, tone: str):
    tone_map = {
        "neutral": "a balanced, neutral tone",
        "facts": "a concise, fact-only tone (no opinions)",
        "simple": "a clear, simple explanation suitable for a 10-year-old"
    }
    tone_instruction = tone_map.get(tone, "a balanced, neutral tone")

    return f"""
Respond ONLY with valid JSON. 
Do not include markdown (```), comments, or extra text.
The JSON must have these keys:
- summary : string ({tone_instruction}, 3–6 sentences)
- keyPoints : array of 5 short strings
- biasScore : number between -50 (left) and +50 (right)
- biasText : short description of bias
- biasPosition : number 0-100 (0=strong left, 50=neutral, 100=strong right)
- tones : object with keys emotional, subjective, positive, negative (0-100 each)

Article:
\"\"\"{article_text}\"\"\"
"""

def clean_and_parse_json(raw_text: str):
    """Cleans Gemini response and extracts valid JSON."""
    raw_clean = raw_text.strip()

    
    if raw_clean.startswith("```"):
        raw_clean = raw_clean.strip("`")
        if raw_clean.lower().startswith("json"):
            raw_clean = raw_clean[4:].strip()

    
    match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if match:
        raw_clean = match.group(0)

    return json.loads(raw_clean)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(force=True)
    content = payload.get("content", "").strip()
    is_url = payload.get("isUrl", False)
    tone = payload.get("tone", "neutral")

    if not content:
        return jsonify({"error": "No content provided"}), 400

    try:
        article_text = extract_text_from_url(content) if is_url else content
    except Exception as e:
        return jsonify({"error": f"URL extraction failed: {str(e)}"}), 400

    prompt = build_prompt(article_text, tone)

    try:
        response = model.generate_content(prompt)
        raw = response.text if hasattr(response, "text") else str(response)

        
        print("RAW GEMINI RESPONSE:\n", raw)

        parsed = clean_and_parse_json(raw)

        
        bias_score = float(parsed.get("biasScore", 0))
        bias_position = float(parsed.get("biasPosition", (bias_score + 50) * 2))
        bias_position = max(0.0, min(100.0, bias_position))

        tones = parsed.get("tones", {})
        def norm(v): 
            try: return max(0, min(100, int(float(v))))
            except: return 0
        tones_norm = {
            "emotional": norm(tones.get("emotional")),
            "subjective": norm(tones.get("subjective")),
            "positive": norm(tones.get("positive")),
            "negative": norm(tones.get("negative")),
        }

        result = {
            "summary": parsed.get("summary", ""),
            "keyPoints": parsed.get("keyPoints", [])[:5],
            "biasScore": bias_score,
            "biasText": parsed.get("biasText", ""),
            "biasPosition": bias_position,
            "tones": tones_norm
        }
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse JSON from Gemini response", "raw": raw}), 500
    except Exception as e:
        return jsonify({"error": f"Gemini call failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
