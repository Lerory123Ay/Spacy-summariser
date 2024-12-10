import spacy
import pytextrank
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load SpaCy model and add PyTextRank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Accepts a POST request with article content and returns summarized content.
    """
    try:
        # Get input article from POST request
        data = request.get_json()
        article_content = data.get("content", "")

        if not article_content:
            return jsonify({"error": "No article content provided"}), 400

        # Process the article with SpaCy and PyTextRank
        doc = nlp(article_content)

        # Generate the summary
        summary_sentences = [
            sent.text for sent in doc._.textrank.summary(limit_phrases=30, limit_sentences=30)
        ]

        # Return summarized content as JSON
        return jsonify({
            "original_content": article_content,
            "summary": " ".join(summary_sentences)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

