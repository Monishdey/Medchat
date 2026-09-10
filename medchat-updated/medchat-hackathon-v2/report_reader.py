"""
Reading a real lab report is harder than it looks.

THE THREE PROBLEMS THIS FILE SOLVES
-----------------------------------

1. WRONG PART OF THE PAGE
   Lab reports print the test names twice: once in the results table, and again
   in the explanatory notes underneath ("TSH is an anterior pituitary hormone
   that..."). Searching the whole page as one lump of text finds whichever comes
   first, which is often the wrong one. So we work LINE BY LINE and only trust
   lines that actually look like a result row.

2. DIFFERENT UNITS
   Indian labs usually report T3 and T4 in ng/mL. The dataset our model learned
   from uses nmol/L. Those are different scales entirely: a T4 of 98 ng/mL is
   126 nmol/L. Feeding 98 straight to the model is simply a wrong number, so we
   detect the unit printed on the line and convert.

3. LOST DECIMAL POINTS
   OCR frequently reads "3.3" as "33". That single missing dot turns a healthy
   patient into a severely ill one. We cannot reliably fix this in code, so
   instead we flag anything that looks physiologically extreme and make the user
   confirm every value before it reaches the model.
"""

import re

# ---------------------------------------------------------------------------
# What each test is called on a real report.
# Order matters: longest names are tried first so "total t4" beats plain "t4".
# ---------------------------------------------------------------------------
TEST_NAMES = {
    "TSH": [
        "thyroid stimulating hormone", "thyroid-stimulating hormone",
        "thyrotropin", "tsh",
    ],
    "T3": [
        "serum triiodothyronine", "triiodothyronine", "total t3", "free t3",
        "ft3", "t3",
    ],
    "TT4": [
        "serum thyroxine", "total thyroxine", "thyroxine", "total t4",
        "free t4", "ft4", "t4",
    ],
    "T4U": ["t4 uptake", "thyroid uptake", "t4u"],
    "FTI": ["free thyroxine index", "ft4i", "fti"],
}

# ---------------------------------------------------------------------------
# Unit conversion into the units our model was trained on.
#
# The model expects:  TSH in mIU/L,  T3 in nmol/L,  TT4 in nmol/L
#
# Molecular weights: T4 = 776.87 g/mol, T3 = 650.98 g/mol. That is where these
# multipliers come from — they are not arbitrary.
# ---------------------------------------------------------------------------
CONVERSIONS = {
    "TSH": {
        "miu/l": 1.0,
        "uiu/ml": 1.0,      # identical scale, just written differently
        "miu/ml": 1000.0,
    },
    "T3": {
        "nmol/l": 1.0,
        "ng/ml": 1.536,     # 1 ng/mL = 1.536 nmol/L
        "ng/dl": 0.01536,
        "pg/ml": 0.001536,
    },
    "TT4": {
        "nmol/l": 1.0,
        "ng/ml": 1.287,     # 1 ng/mL = 1.287 nmol/L
        "ug/dl": 12.87,     # 1 µg/dL = 12.87 nmol/L
        "mcg/dl": 12.87,
        "ng/dl": 0.01287,
    },
}

# OCR mangles units badly: "ng/mL" becomes "ngimL", "ngimt", "ng!mL".
# So we do not match units exactly. We look for the pieces.
def detect_unit(text: str) -> str | None:
    """
    Work out the unit from the short piece of text right after the number.

    Pass only a small window. Given a whole line, the "ng" inside a word like
    "stimulating" would be mistaken for nanograms.
    """
    t = text.lower().replace(" ", "")
    has = lambda *bits: any(b in t for b in bits)

    # µIU/mL survives OCR as plU, wlU, ulU, piU, wIU and various others.
    if has("uiu", "µiu", "piu", "plu", "wlu", "wiu", "ulu", "iu/m", "iu/e"):
        return "uiu/ml"
    if has("miu"):
        return "miu/l"
    if has("nmol"):
        return "nmol/l"
    if has("pmol"):
        return "pmol/l"
    if has("ng"):
        if has("/dl", "idl", "!dl"):
            return "ng/dl"
        return "ng/ml"       # ngimL, ngimt, ng/mL all land here
    if has("ug", "mcg", "µg"):
        return "ug/dl"
    return None


# Physiologically plausible bounds, AFTER conversion to model units.
# Anything outside these is almost certainly an OCR error, not a real patient.
PLAUSIBLE = {
    "TSH": (0.01, 150.0),
    "T3": (0.1, 12.0),
    "TT4": (5.0, 400.0),
    "T4U": (0.3, 2.0),
    "FTI": (5.0, 400.0),
}

# Values this far outside the normal range are possible but unusual, so we ask
# the user to double check rather than silently trusting the OCR.
SUSPICIOUS = {
    "TSH": (0.1, 20.0),
    "T3": (0.5, 5.0),
    "TT4": (30.0, 200.0),
}


def looks_like_prose(line: str) -> bool:
    """Explanatory paragraphs have lots of words and few numbers."""
    words = line.split()
    if len(words) > 12:
        return True
    # A results row always has at least one number in it.
    return not re.search(r"\d", line)


def numbers_in(text: str) -> list[float]:
    """
    Every number in a piece of text, in order.

    The lookbehind matters more than it looks. Test codes contain digits: the
    "3" in "T3" and the "4" in "T4" are part of the NAME, not the result. So we
    skip any digit that has a letter directly in front of it. Without this,
    "SERUM TRIIODOTHYRONINE, T3 1.86" reads as 3 instead of 1.86.
    """
    out = []
    for raw in re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?", text):
        try:
            out.append(float(raw))
        except ValueError:
            pass
    return out


def parse_report(text: str) -> dict:
    """
    Read a lab report line by line.

    Returns a dict like:
        {
          "values": {"TSH": 3.3, "T3": 2.86, ...},
          "details": [ {field, raw_value, unit, converted, warning}, ... ],
          "warnings": ["..."],
        }
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    values: dict[str, float] = {}
    details: list[dict] = []
    warnings: list[str] = []

    for line in lines:
        if looks_like_prose(line):
            continue

        low = line.lower()
        # OCR turns the I in FTI into a pipe or a one.
        low = re.sub(r"\bft[|!l1](?![a-z0-9])", "fti", low)

        for field, names in TEST_NAMES.items():
            if field in values:
                continue  # already found it on an earlier line

            for name in names:
                position = low.find(name)
                if position == -1:
                    continue

                # Look at what comes AFTER the test name on this line.
                after = line[position + len(name):]
                found_numbers = numbers_in(after)
                if not found_numbers:
                    continue

                raw_value = found_numbers[0]

                # Only look at the ~16 characters immediately after the number
                # for the unit. Scanning the whole line would pick up the "ng"
                # buried inside a word like "stimulating".
                value_text = re.search(
                    r"(?<![A-Za-z0-9])" + re.escape(f"{raw_value:g}"), after
                )
                window = after[value_text.end(): value_text.end() + 16] if value_text else after[:16]
                unit = detect_unit(window)

                # Convert into the units the model was trained on.
                converted = raw_value
                if unit and field in CONVERSIONS:
                    factor = CONVERSIONS[field].get(unit)
                    if factor:
                        converted = round(raw_value * factor, 3)
                    elif unit == "pmol/l":
                        warnings.append(
                            f"{field} is reported in pmol/L (a free-hormone unit). "
                            f"This model was trained on total-hormone values, so "
                            f"{field} was left out rather than converted wrongly."
                        )
                        break

                # Reject anything physically impossible.
                low_bound, high_bound = PLAUSIBLE[field]
                if not (low_bound <= converted <= high_bound):
                    warnings.append(
                        f"{field} read as {raw_value}"
                        f"{' ' + unit if unit else ''} which is outside any "
                        f"realistic range. Ignored — please type it in manually."
                    )
                    break

                values[field] = converted
                detail = {
                    "field": field,
                    "raw": raw_value,
                    "unit": unit or "assumed model units",
                    "value": converted,
                    "converted": converted != raw_value,
                }

                # Flag values that are possible but unusual, because a lost
                # decimal point is the most common OCR failure.
                if field in SUSPICIOUS:
                    lo, hi = SUSPICIOUS[field]
                    if not (lo <= converted <= hi):
                        detail["check"] = True
                        warnings.append(
                            f"{field} came out as {converted}. That is possible, "
                            f"but OCR often drops a decimal point. Please check it "
                            f"against your report before trusting the result."
                        )

                details.append(detail)
                break

    # ---- age and sex -------------------------------------------------------
    # Reports write this as "Age / Sex : 42 YRS / M" or "Age: 42  Sex: M".
    age_match = (
        re.search(r"age\s*[/\\]?\s*sex\s*[:\-]?\s*(\d{1,3})", text, re.I)
        or re.search(r"\bage\b\s*[:\-]?\s*(\d{1,3})", text, re.I)
        or re.search(r"\b(\d{1,3})\s*(?:yrs?|years?)\b", text, re.I)
    )
    if age_match:
        age = int(age_match.group(1))
        if 0 < age < 120:
            values["age"] = float(age)

    # Sex is usually the lone M or F right after the age.
    sex_match = (
        re.search(r"\b\d{1,3}\s*(?:yrs?|years?)?\s*[/\\]\s*([mf])\b", text, re.I)
        or re.search(r"\bsex\s*[:\-]?\s*([mf])\b", text, re.I)
        or re.search(r"\bsex\s*[:\-]?\s*(male|female)\b", text, re.I)
    )
    if sex_match:
        token = sex_match.group(1).lower()
        values["sex"] = 1.0 if token in ("m", "male") else 0.0

    # ---- flags mentioned anywhere on the page ------------------------------
    low_all = text.lower()
    for flag, words in {
        "on_thyroxine": ["levothyroxine", "eltroxin", "thyronorm", "on thyroxine"],
        "pregnant": ["pregnant", "pregnancy", "antenatal"],
        "goitre": ["goitre", "goiter"],
        "thyroid_surgery": ["thyroidectomy", "thyroid surgery"],
    }.items():
        if any(word in low_all for word in words):
            values[flag] = 1.0

    return {"values": values, "details": details, "warnings": warnings}
