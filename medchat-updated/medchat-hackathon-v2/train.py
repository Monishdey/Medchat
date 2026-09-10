"""
MedChat — train the T1 thyroid model.

Run it with:   python train.py

What this file does, in order:
  1. Reads the patient spreadsheet (data/hypothyroid.csv)
  2. Cleans it up so a computer can learn from it
  3. Splits it into "study material" and "exam questions"
  4. Trains a Random Forest
  5. Scores the model and prints a report card
  6. Saves the trained model to model/ so app.py can use it

Every step is explained in the README. Read that alongside this file.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

HERE = Path(__file__).parent
DATA = HERE / "data" / "hypothyroid.csv"
OUT = HERE / "model"

# ---------------------------------------------------------------------------
# STEP 0 — Which columns do we feed the model?
# ---------------------------------------------------------------------------
# These 6 are numbers (blood test results + age).
NUMBERS = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

# These are yes/no answers. In the file they are stored as the letters t and f.
YESNO = [
    "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment",
    "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
    "tumor", "hypopituitary", "psych",
]

FEATURES = NUMBERS + YESNO

# The three answers the model is allowed to give.
CLASSES = ["negative", "compensated_hypothyroid", "primary_hypothyroid"]

# Friendly names, used later in the chat window.
NICE_NAMES = {
    "negative": "No hypothyroidism detected",
    "compensated_hypothyroid": "Compensated (early / subclinical) hypothyroidism",
    "primary_hypothyroid": "Primary hypothyroidism",
}


def load_and_clean():
    """Turn the messy CSV into a clean table of numbers."""
    # na_values="?" tells pandas: wherever you see a question mark, that means
    # "this test was never done", not the literal text "?".
    df = pd.read_csv(DATA, na_values=["?"])
    print(f"Loaded {len(df)} patient records with {len(df.columns)} columns.")

    # --- Drop columns that would let the model cheat -----------------------
    # The "_measured" columns say whether a doctor ordered each test. A doctor
    # only orders a thyroid test when they already suspect thyroid disease, so
    # these columns secretly contain the doctor's opinion. If we left them in,
    # the model would learn "doctor ordered the test => patient is ill" instead
    # of learning any actual medicine. This is called DATA LEAKAGE.
    leaky = [c for c in df.columns if c.endswith("_measured")]
    # TBG is empty for 96% of patients, so it teaches us nothing.
    df = df.drop(columns=leaky + ["TBG"], errors="ignore")
    print(f"Dropped {len(leaky) + 1} columns that would cause data leakage.")

    # --- Keep only the three classes we can actually learn -----------------
    # There is a 4th class, secondary_hypothyroid, but only 2 patients have it.
    # You cannot teach anything from 2 examples, so we remove them and say so.
    before = len(df)
    df = df[df["Class"].isin(CLASSES)].copy()
    print(f"Removed {before - len(df)} records from a class too rare to learn.")

    # --- Convert letters into numbers --------------------------------------
    # Computers do maths, not letters. t/f becomes 1/0, M/F becomes 1/0.
    for col in YESNO:
        if col == "sex":
            df[col] = df[col].map({"M": 1, "F": 0})
        else:
            df[col] = df[col].map({"t": 1, "f": 0})

    for col in NUMBERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Fix impossible ages ------------------------------------------------
    # A few rows say the patient is 455 years old. Typing mistakes from 1987.
    # We mark them as "unknown" rather than trusting them.
    bad_ages = ((df["age"] > 120) | (df["age"] < 0)).sum()
    df.loc[(df["age"] > 120) | (df["age"] < 0), "age"] = np.nan
    print(f"Fixed {bad_ages} impossible ages.")

    return df


def main():
    OUT.mkdir(exist_ok=True)
    df = load_and_clean()

    # X = the questions (patient details). y = the answers (the diagnosis).
    X = df[FEATURES]
    y = df["Class"]

    print("\nHow many patients of each type:")
    for name, count in y.value_counts().items():
        print(f"  {name:<26} {count:>5}  ({count / len(y) * 100:.1f}%)")

    # -----------------------------------------------------------------------
    # STEP 1 — Split the data
    # -----------------------------------------------------------------------
    # 80% to learn from, 20% hidden away to test with. Testing on data the
    # model already saw is like giving a student the exam paper to revise from:
    # the score means nothing.
    #
    # stratify=y keeps the same mix of ill/healthy patients in both halves.
    # random_state=42 makes the split identical every run, so your results
    # are reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTraining on {len(X_train)} patients, testing on {len(X_test)} unseen ones.")

    # -----------------------------------------------------------------------
    # STEP 2 — Build the model
    # -----------------------------------------------------------------------
    # A Pipeline glues steps together so the same cleaning happens at training
    # time AND at prediction time. Forgetting this is the #1 hackathon bug.
    #
    #   SimpleImputer: fills blank test results with the median value.
    #   RandomForest:  400 decision trees that vote on the answer.
    #
    # class_weight="balanced" matters a lot here. 92% of patients are healthy,
    # so a lazy model could answer "healthy" every time and be 92% accurate
    # while being completely useless. This setting makes mistakes on rare
    # classes cost more, forcing the model to actually pay attention to them.
    model = Pipeline([
        ("fill_blanks", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # -----------------------------------------------------------------------
    # STEP 3 — Cross-validation (a practice exam)
    # -----------------------------------------------------------------------
    # Splits the training data 5 ways and trains 5 times. If all 5 scores are
    # close together, the model is stable. If they swing wildly, your single
    # test score was luck.
    print("\nRunning 5-fold cross-validation (this trains the model 5 times)...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"  Scores per fold: {[round(s, 3) for s in scores]}")
    print(f"  Average: {scores.mean():.4f}  (spread +/- {scores.std():.4f})")

    # -----------------------------------------------------------------------
    # STEP 4 — Train for real, then take the exam
    # -----------------------------------------------------------------------
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n" + "=" * 62)
    print("REPORT CARD  (on the 20% of patients the model never saw)")
    print("=" * 62)
    print(classification_report(y_test, predictions, digits=4))

    print("Confusion matrix — rows are the truth, columns are the guess:")
    cm = pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=model.classes_),
        index=[f"true {c}" for c in model.classes_],
        columns=[f"said {c}" for c in model.classes_],
    )
    print(cm.to_string())

    macro_f1 = f1_score(y_test, predictions, average="macro")

    # -----------------------------------------------------------------------
    # STEP 5 — What did the model learn to care about?
    # -----------------------------------------------------------------------
    # If the top features are not the ones a doctor would name, something is
    # wrong. Here TSH should come first. This is your sanity check.
    importances = sorted(
        zip(FEATURES, model.named_steps["forest"].feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )
    print("\nWhat the model pays most attention to:")
    for name, score in importances[:6]:
        bar = "#" * int(score * 60)
        print(f"  {name:<10} {score:.4f}  {bar}")

    # -----------------------------------------------------------------------
    # STEP 6 — Save everything
    # -----------------------------------------------------------------------
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "classes": list(model.classes_),
        "nice_names": NICE_NAMES,
    }, OUT / "t1_model.joblib")

    scorecard = {
        "macro_f1": round(float(macro_f1), 4),
        "cv_mean": round(float(scores.mean()), 4),
        "cv_std": round(float(scores.std()), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "top_features": [
            {"name": n, "importance": round(float(s), 4)} for n, s in importances[:6]
        ],
        "per_class": classification_report(y_test, predictions, output_dict=True),
    }
    (OUT / "t1_scorecard.json").write_text(json.dumps(scorecard, indent=2))

    print(f"\nSaved the model to  {OUT / 't1_model.joblib'}")
    print(f"Saved the scorecard to {OUT / 't1_scorecard.json'}")
    print("\nDone. Now run:  python app.py")


if __name__ == "__main__":
    main()
