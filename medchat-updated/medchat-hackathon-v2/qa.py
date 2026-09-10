"""
MedChat's answering brain. No AI language model, no API key, no internet.

THE IDEA
--------
We have ~32 written question-and-answer pairs in data/faq.json. When you ask
something, we need to find which of those 32 is closest to your question.

We do it with TF-IDF plus cosine similarity:

  TF-IDF  turns every sentence into a list of numbers. Words that are rare
          across the collection (like "levothyroxine") get a high score, and
          words that appear everywhere (like "the" or "thyroid") get a low one,
          because common words do not help you tell entries apart.

  Cosine  measures the angle between two of those number-lists. Identical
  similarity  meaning gives 1.0, nothing in common gives 0.0.

So "what pill do I take for an underactive thyroid" scores highly against the
levothyroxine entry, even though it shares almost no exact words with it.

WHY NOT AN AI MODEL?
--------------------
Three reasons that matter for a hackathon:
  1. It is free and needs no key, so it cannot stop working during your demo.
  2. It cannot hallucinate. Every answer is one a human wrote and checked,
     which is the right property for anything health related.
  3. It answers in about 1 millisecond.

The trade-off is honest: it can only answer what is already written down.
When nothing matches well enough, it says so instead of making something up.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = Path(__file__).parent / "data" / "faq.json"

# If the best match scores below this, we admit we do not know.
# Lower it and you get confident nonsense; raise it and it refuses too often.
# 0.08 was picked by trying real questions and watching what happened.
MIN_SCORE = 0.08


class ThyroidQA:
    def __init__(self, faq_path: Path = FAQ_PATH):
        data = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.entries = data["faq"]

        # We search TWO things separately and then blend the scores.
        #
        # Why not one big blob? Because answers are long and questions are
        # short. Mixed together, the long answers drown out the questions and
        # matching gets worse. Keeping them apart lets us say "a hit on the
        # title is worth more than a hit buried in the body".
        titles = [e["q"] + " " + " ".join(e.get("tags", [])) for e in self.entries]
        bodies = [e["a"] for e in self.entries]

        def build(corpus):
            vec = TfidfVectorizer(
                stop_words="english",   # ignore "the", "is", "a" and friends
                ngram_range=(1, 2),     # index single words AND two-word phrases
                sublinear_tf=True,      # stop long text from dominating
                min_df=1,
            )
            return vec, vec.fit_transform(corpus)

        self.title_vec, self.title_matrix = build(titles)
        self.body_vec, self.body_matrix = build(bodies)

    def _scores(self, question: str):
        """Blend: a match on the question line counts double a body match."""
        title_scores = cosine_similarity(
            self.title_vec.transform([question]), self.title_matrix
        )[0]
        body_scores = cosine_similarity(
            self.body_vec.transform([question]), self.body_matrix
        )[0]
        return 0.7 * title_scores + 0.3 * body_scores

    def search(self, question: str, top_n: int = 3) -> list[dict]:
        """Return the closest entries, best first."""
        if not question.strip():
            return []

        scores = self._scores(question)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        results = []
        for index, score in ranked[:top_n]:
            if score < MIN_SCORE:
                continue
            results.append({
                "question": self.entries[index]["q"],
                "answer": self.entries[index]["a"],
                "score": round(float(score), 3),
            })
        return results

    def answer(self, question: str) -> dict:
        """Answer a question, or admit defeat gracefully."""
        hits = self.search(question, top_n=3)

        if not hits:
            return {
                "found": False,
                "answer": (
                    "I do not have a written answer for that one. My knowledge "
                    "base covers thyroid basics, blood test meanings, symptoms, "
                    "treatment and how this app works. Try rephrasing, or ask "
                    "me to assess a set of lab values instead."
                ),
                "related": [],
            }

        return {
            "found": True,
            "answer": hits[0]["answer"],
            "matched_question": hits[0]["question"],
            "score": hits[0]["score"],
            # Show the runners-up as clickable suggestions in the UI.
            "related": [h["question"] for h in hits[1:]],
        }

    def sample_questions(self, n: int = 4) -> list[str]:
        return [e["q"] for e in self.entries[:n]]


if __name__ == "__main__":
    # Quick manual test:  python qa.py
    qa = ThyroidQA()
    for question in [
        "what pill do I take for an underactive thyroid",
        "is my tsh of 8 bad",
        "can I still have a baby",
        "how do you know this is accurate",
        "what is the capital of France",
    ]:
        result = qa.answer(question)
        label = result.get("matched_question", "NO MATCH")
        print(f"\nQ: {question}\n-> [{label}]\n   {result['answer'][:110]}...")
