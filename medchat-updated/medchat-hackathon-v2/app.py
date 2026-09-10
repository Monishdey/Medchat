"""
MedChat — the whole backend in one file.

Start it with:   python app.py
Then open:       http://localhost:8000

WHAT HAPPENS WHEN YOU SEND A MESSAGE
------------------------------------
1. We scan your text for lab values (TSH, T4, etc).
2. If we found TSH plus at least one more value  -> run the T1 model.
   Otherwise                                     -> search the Q&A knowledge base.
3. Send the answer back to the browser.

That decision in step 2 is called INTENT ROUTING. It is the entire "brain" of
the chat, and it is about 10 lines of code. No AI language model involved.
"""

import io
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import doctor
from qa import ThyroidQA
from report_reader import parse_report

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load the trained model and the Q&A brain once, when the server starts.
# Loading them on every request would make the app painfully slow.
# ---------------------------------------------------------------------------
bundle = joblib.load(HERE / "model" / "t1_model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]
NICE_NAMES = bundle["nice_names"]
SCORECARD = json.loads((HERE / "model" / "t1_scorecard.json").read_text())
QA = ThyroidQA()

# ---------------------------------------------------------------------------
# Normal ranges. These are ONLY used to write the explanation in plain English.
# The model itself never sees them — it learned its own thresholds from data.
# ---------------------------------------------------------------------------
NORMAL_RANGES = {
    "TSH": (0.4, 4.0, "mIU/L"),
    "T3": (0.9, 2.5, "nmol/L"),
    "TT4": (60.0, 140.0, "nmol/L"),
    "T4U": (0.7, 1.3, ""),
    "FTI": (60.0, 155.0, ""),
}

# All the ways people and lab reports write each test.
NAME_VARIANTS = {
    "TSH": ["tsh", "thyroid stimulating hormone", "thyrotropin"],
    "T3": ["t3", "free t3", "ft3", "triiodothyronine"],
    "TT4": ["tt4", "total t4", "t4", "free t4", "ft4", "thyroxine"],
    "T4U": ["t4u", "t4 uptake", "thyroid uptake"],
    "FTI": ["fti", "free thyroxine index", "ft4i"],
}


def extract_values(text: str) -> dict:
    """
    Pull lab numbers out of ordinary writing.

    Handles all of these:
        "TSH 14.2, TT4 48"
        "46F, TSH 14.2"
        "Thyroid Stimulating Hormone .... 14.2 mIU/L"
        "thyroid stimulating hormone: 14.2"
    """
    found = {}
    low = text.lower()

    # Tesseract misreads the capital I in "FTI" as a pipe or a 1 on photos.
    low = re.sub(r"\bft[|!l1](?![a-z0-9])", "fti", low)

    for field, variants in NAME_VARIANTS.items():
        # Try the longest name first, so "total t4" wins over plain "t4".
        for name in sorted(variants, key=len, reverse=True):
            # Look for the test name, then skip over any colons, dots or units,
            # then grab the first number.
            pattern = rf"\b{re.escape(name)}\b[^0-9\-\n]{{0,20}}?(\d+(?:\.\d+)?)"
            match = re.search(pattern, low)
            if match:
                found[field] = float(match.group(1))
                break

    # Age written out: "46 years old", "age: 46"
    match = re.search(r"\b(\d{1,3})\s*[- ]?\s*year[s]?[- ]?old\b", low) or \
            re.search(r"\bage\D{0,6}(\d{1,3})\b", low)
    if match and 0 < int(match.group(1)) < 120:
        found["age"] = float(match.group(1))

    # Clinical shorthand: "46F", "62 M", "F/33"
    match = re.search(r"\b(\d{1,3})\s*[/ ]?\s*([mf])\b|\b([mf])\s*[/ ]?\s*(\d{1,3})\b", low)
    if match:
        age = match.group(1) or match.group(4)
        sex = match.group(2) or match.group(3)
        if age and 0 < int(age) < 120:
            found.setdefault("age", float(age))
            found.setdefault("sex", 1.0 if sex == "m" else 0.0)

    # Words for sex
    if "sex" not in found:
        match = re.search(r"\b(male|female|man|woman|boy|girl)\b", low)
        if match:
            found["sex"] = 1.0 if match.group(1) in ("male", "man", "boy") else 0.0

    # Yes/no flags mentioned in passing
    flag_words = {
        "on_thyroxine": ["on thyroxine", "levothyroxine", "eltroxin", "on t4"],
        "pregnant": ["pregnant", "pregnancy"],
        "sick": ["currently sick", "acutely unwell"],
        "goitre": ["goitre", "goiter"],
        "thyroid_surgery": ["thyroid surgery", "thyroidectomy"],
        "tumor": ["nodule", "thyroid tumor", "thyroid tumour"],
    }
    for flag, words in flag_words.items():
        if any(word in low for word in words):
            found[flag] = 1.0

    return found


def predict(values: dict) -> dict:
    """Run the T1 model on whatever values we have."""
    # The model was trained on 21 columns and expects all 21, in order.
    # Anything the user did not give us is left blank, and the SimpleImputer
    # inside the pipeline fills it with the median it learned during training.
    row = {feature: values.get(feature) for feature in FEATURES}
    frame = pd.DataFrame([row])[FEATURES]

    probabilities = MODEL.predict_proba(frame)[0]
    ranked = sorted(
        zip(MODEL.classes_, probabilities), key=lambda pair: pair[1], reverse=True
    )
    best_class, best_probability = ranked[0]

    # Build a plain-English explanation from the reference ranges.
    reasons = []
    for test, (low, high, unit) in NORMAL_RANGES.items():
        if values.get(test) is None:
            continue
        value = values[test]
        if value > high:
            reasons.append(f"{test} is {value} {unit}, above the usual {low}–{high}.".replace("  ", " "))
        elif value < low:
            reasons.append(f"{test} is {value} {unit}, below the usual {low}–{high}.".replace("  ", " "))
    if values.get("on_thyroxine"):
        reasons.append("You mentioned thyroxine, which changes how these numbers read.")
    if values.get("pregnant"):
        reasons.append("Pregnancy shifts thyroid ranges. Please see a doctor about this.")
    if not reasons:
        reasons.append("Every value you gave sits inside its usual range.")

    important = ["age", "sex", "TSH", "T3", "TT4", "T4U", "FTI"]
    missing = [f for f in important if values.get(f) is None]

    return {
        "type": "prediction",
        "label": NICE_NAMES.get(best_class, best_class),
        "confidence": round(float(best_probability), 4),
        "probabilities": [
            {"label": NICE_NAMES.get(c, c), "value": round(float(p), 4)}
            for c, p in ranked
        ],
        "values_used": {
            k: ("male" if v == 1 else "female") if k == "sex" else v
            for k, v in values.items() if v is not None
        },
        "missing": missing,
        "reasons": reasons,
        # Doctor mode uses these two; patient mode ignores them.
        "flags": doctor.flag_reasons(values, {
            "confidence": round(float(best_probability), 4),
            "missing": missing,
        }),
        "note": None,   # filled in by chat() only when mode == "doctor"
    }


# ---------------------------------------------------------------------------
# The web server
# ---------------------------------------------------------------------------
app = FastAPI(title="MedChat")


class ChatMessage(BaseModel):
    message: str
    model_id: str = "T1"
    mode: str = "patient"          # "patient" or "doctor"


@app.post("/api/chat")
def chat(payload: ChatMessage):
    """The intent router. This is the heart of the app."""
    if payload.model_id != "T1":
        return {
            "type": "text",
            "answer": "That model is not built yet. Switch back to T1 (Thyroid).",
        }

    values = extract_values(payload.message)
    lab_tests = [k for k in ("TSH", "T3", "TT4", "T4U", "FTI") if k in values]

    # RULE: we need TSH plus one more test before we will guess at anything.
    if "TSH" in values and len(lab_tests) >= 2:
        result = predict(values)
        if payload.mode == "doctor":
            # Clinicians get the paste-into-records block. Patients do not,
            # because it is written in clinical register and would confuse.
            result["note"] = doctor.clinical_note(values, result)
        return result

    # We found something, but not enough. Ask for the rest instead of guessing.
    if lab_tests:
        still_need = [t for t in ("TSH", "TT4", "FTI") if t not in values]
        return {
            "type": "text",
            "answer": (
                f"I can see {', '.join(lab_tests)}. To run the model I also need "
                f"{' and '.join(still_need)} from the same blood test. "
                "Age and sex make it more accurate too."
            ),
        }

    # No lab values at all, so treat it as a question.
    result = QA.answer(payload.message)
    answer = result["answer"]

    if payload.mode == "doctor" and result.get("found"):
        # The knowledge base is written for patients. Say so rather than
        # letting a clinician assume it is a clinical reference.
        answer += ("\n\nNote: this knowledge base is patient-education material, "
                   "not a clinical reference. For guideline detail see ATA/BTA.")

    return {
        "type": "text",
        "answer": answer,
        "source": result.get("matched_question"),
        "related": result.get("related", []),
    }


class TriageRequest(BaseModel):
    text: str


@app.post("/api/triage")
def triage_batch(payload: TriageRequest):
    """
    DOCTOR MODE. Take many patients at once and return them ranked.

    Input is one patient per line, optionally with an identifier:

        P001: 46F TSH 14.2 TT4 48 T3 1.0 FTI 51
        P002: 30M TSH 1.8 TT4 110 T3 2.0 FTI 110
        Bed 7 - 62F TSH 9.1 FTI 70

    Output is the same list sorted so the patients needing attention first
    are at the top.
    """
    lines = doctor.split_patients(payload.text)

    if not lines:
        return {"type": "text", "answer": "No patient rows found. One patient per line."}

    if len(lines) > 100:
        return {"type": "text", "answer": "Batch limit is 100 rows. Split the list."}

    rows = []
    skipped = []

    for index, line in enumerate(lines):
        label = doctor.patient_label(line, index)
        values = extract_values(doctor.strip_label(line, label))

        lab_tests = [k for k in ("TSH", "T3", "TT4", "T4U", "FTI") if k in values]
        if "TSH" not in values or len(lab_tests) < 2:
            # Same refusal rule as patient mode. We do not guess in bulk either.
            skipped.append({"label": label, "line": line, "found": lab_tests})
            continue

        result = predict(values)
        rows.append({
            "label": label,
            "class_id": [c for c in MODEL.classes_
                         if NICE_NAMES.get(c, c) == result["label"]][0],
            "prediction": result["label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "values": result["values_used"],
            "missing": result["missing"],
            "flags": doctor.flag_reasons(values, result),
            "note": doctor.clinical_note(values, result, label),
        })

    return {
        "type": "triage",
        "total": len(lines),
        "assessed": len(rows),
        "skipped": skipped,
        "worklist": doctor.triage(rows),
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Read a photo of a lab report and pull the numbers off it."""
    raw = await file.read()

    if len(raw) > 8 * 1024 * 1024:
        return JSONResponse({"error": "Image must be under 8 MB."}, status_code=413)

    try:
        import pytesseract
        from PIL import Image, ImageOps
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except ImportError:
        return JSONResponse(
            {"error": "Image reading needs Tesseract. See the README, step 2."},
            status_code=501,
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image = ImageOps.exif_transpose(image).convert("L")  # rotate + greyscale
        # Small phone photos read badly, so enlarge them first.
        if min(image.size) < 1000:
            factor = 1000 / min(image.size)
            image = image.resize((int(image.width * factor), int(image.height * factor)))
        text = pytesseract.image_to_string(image)
    except Exception as error:
        return JSONResponse({"error": f"Could not read that image: {error}"}, status_code=400)

    values = parse_report(text)
    shown = {k: v for k, v in values["values"].items() if k != "sex"}
    if "sex" in values["values"]:
        shown["sex"] = "M" if values["values"]["sex"] == 1.0 else "F"

    return {
        "filename": file.filename,
        "text": text.strip(),
        "found": shown,
        "details": values["details"],
        "warnings": values["warnings"],
        "count": len(values["values"]),
    }


@app.get("/api/models")
def models():
    """Fills the model switcher menu in the UI."""
    return {"models": [
        {
            "id": "T1",
            "name": "MedChat T1",
            "area": "Thyroid",
            "note": "Hypothyroidism screening from a blood panel",
            "ready": True,
        },
        {
            "id": "C1",
            "name": "MedChat C1",
            "area": "Cancer",
            "note": "Thyroid cancer recurrence risk",
            "ready": False,
        },
    ]}


@app.get("/api/scorecard")
def scorecard():
    return SCORECARD


@app.get("/")
def home():
    return FileResponse(HERE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("  MedChat is starting")
    print("  Open this in your browser:  http://localhost:8000")
    print("  Press CTRL+C to stop")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
