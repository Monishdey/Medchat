"""
DOCTOR MODE — the clinician-facing side of MedChat.

WHY THIS EXISTS
---------------
A patient and a doctor want opposite things from the same model.

  A patient has ONE set of results and needs them explained gently.
  A doctor has FIFTY sets of results and needs to know who to look at FIRST.

So doctor mode is not a re-skin of patient mode. It is a different tool built
on the same model:

  * BATCH TRIAGE — paste many patients at once, get them back ranked by
    priority, so the sickest are at the top of the list.
  * CLINICAL NOTE — one-click text to paste into patient records.
  * NO SOFTENING — raw probabilities, every missing field named, no
    reassurance language.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not diagnose, prescribe, or replace review. It sorts a queue. The
doctor still opens every case. A tool that told a clinician "you can skip
this one" would be dangerous; a tool that says "start with this one" is not.
"""

from __future__ import annotations

import re

# Priority bands. These decide the order of the worklist.
#
# Note the ordering logic: we sort by PRIORITY first, and only then by the
# model's confidence. A high-confidence "normal" must never outrank a
# low-confidence "possible primary hypothyroidism".
PRIORITY = {
    "primary_hypothyroid": 3,
    "compensated_hypothyroid": 2,
    "negative": 1,
}

BAND_LABEL = {3: "Review first", 2: "Review", 1: "Routine"}
BAND_KEY = {3: "high", 2: "medium", 1: "low"}


def split_patients(text: str) -> list[str]:
    """
    One patient per line.

    Blank lines are skipped. Lines beginning with # are treated as comments,
    so a doctor can paste a list with headings in it.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def patient_label(line: str, index: int) -> str:
    """
    Pull an identifier off the start of the line if there is one.

    Accepts "P001: 46F TSH 14.2" or "Bed 7 - 46F TSH 14.2" and returns the
    bit before the separator. Falls back to a row number.
    """
    match = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9 _/#-]{0,24}?)\s*[:\-|]\s+", line)
    if match:
        label = match.group(1).strip()
        # Guard against swallowing an actual lab value, e.g. "TSH 14.2 - low"
        if not re.search(r"\d+\.\d", label):
            return label
    return f"Row {index + 1}"


def strip_label(line: str, label: str) -> str:
    """Remove the identifier so the value parser does not read it as data."""
    if label.startswith("Row "):
        return line
    return re.sub(r"^\s*" + re.escape(label) + r"\s*[:\-|]\s+", "", line)


def flag_reasons(values: dict, result: dict) -> list[str]:
    """
    Things a clinician would want flagged, beyond the classification itself.

    These are the situations where the model's output should be treated with
    extra suspicion, or where the guidelines say something specific.
    """
    flags = []
    tsh = values.get("TSH")

    if tsh is not None:
        if tsh >= 50:
            flags.append("TSH >= 50 — marked elevation")
        elif tsh >= 10:
            flags.append("TSH >= 10 — treatment usually considered")
        elif tsh < 0.1:
            flags.append("TSH suppressed — consider hyperthyroidism, outside T1 scope")

    if values.get("pregnant"):
        flags.append("Pregnancy — trimester-specific ranges apply, model not valid")
    if values.get("on_thyroxine"):
        flags.append("On thyroxine — raised TSH may indicate under-replacement")
    if values.get("sick"):
        flags.append("Acute illness — non-thyroidal illness may distort results")
    if values.get("thyroid_surgery"):
        flags.append("Prior thyroid surgery")

    missing = result.get("missing", [])
    if len(missing) >= 3:
        flags.append(f"Sparse data — {len(missing)} of 7 inputs absent")

    if result["confidence"] < 0.6:
        flags.append("Low model confidence — weak separation between classes")

    return flags


def triage(rows: list[dict]) -> list[dict]:
    """
    Sort a list of already-predicted patients into a worklist.

    Sort order, in strict precedence:
      1. Priority band  (ill outranks well, always)
      2. Any clinical flags raised
      3. Model confidence within the band
    """
    for row in rows:
        band = PRIORITY.get(row["class_id"], 1)
        row["priority"] = band
        row["band"] = BAND_LABEL[band]
        row["band_key"] = BAND_KEY[band]

    return sorted(
        rows,
        key=lambda r: (r["priority"], len(r["flags"]), r["confidence"]),
        reverse=True,
    )


def clinical_note(values: dict, result: dict, label: str = "") -> str:
    """
    A plain-text block a doctor can paste straight into a record.

    Deliberately written so that anyone reading the notes later can see this
    came from a screening model and knows exactly what went into it.
    """
    def fmt(key):
        v = values.get(key)
        if v is None:
            return "not supplied"
        if key == "sex":
            return "M" if v == 1 else "F"
        if key == "age":
            return str(int(v))
        return str(v)

    lines = [
        "MedChat T1 screening output — decision support only, not a diagnosis.",
        "",
        f"Patient: {label or 'unspecified'}",
        f"Age {fmt('age')}, sex {fmt('sex')}",
        "",
        "Inputs (converted to nmol/L where applicable):",
        f"  TSH {fmt('TSH')} mIU/L",
        f"  T3  {fmt('T3')}",
        f"  TT4 {fmt('TT4')}",
        f"  T4U {fmt('T4U')}",
        f"  FTI {fmt('FTI')}",
        "",
        f"Model output: {result['label']} ({result['confidence'] * 100:.1f}% confidence)",
    ]

    for probability in result["probabilities"]:
        lines.append(f"  {probability['label']}: {probability['value'] * 100:.1f}%")

    if result.get("missing"):
        lines += [
            "",
            "Not supplied, dataset median substituted: " + ", ".join(result["missing"]),
        ]

    flags = flag_reasons(values, result)
    if flags:
        lines += ["", "Flags:"] + [f"  - {f}" for f in flags]

    lines += [
        "",
        "Model: random forest, 400 trees, trained on 3,016 records from the UCI",
        "Thyroid Disease dataset (Garavan Institute). Held-out macro F1 0.9868.",
        "Historical single-centre data. Not validated prospectively. Not a",
        "medical device. Clinical correlation required.",
    ]
    return "\n".join(lines)
