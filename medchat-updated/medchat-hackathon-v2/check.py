"""
Quick self-check. Run this before your demo:   python check.py

It confirms the model file exists, the model predicts sensibly, the Q&A search
finds the right answers, and the server routes all respond. If every line says
PASS, your demo will work.
"""
import sys
from pathlib import Path

results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    mark = "  PASS" if condition else "  FAIL"
    print(f"{mark}  {name}" + (f"   ({detail})" if detail else ""))


print("\nChecking MedChat...\n")

# 1. Is the model trained yet?
model_file = Path(__file__).parent / "model" / "t1_model.joblib"
check("Model file exists", model_file.exists(),
      "" if model_file.exists() else "fix: run  python train.py")
if not model_file.exists():
    sys.exit(1)

from fastapi.testclient import TestClient          # noqa: E402
from app import app, extract_values                # noqa: E402
from qa import ThyroidQA                           # noqa: E402

client = TestClient(app)

# 2. Does it read lab values out of ordinary writing?
values = extract_values("46F, TSH 14.2, TT4 48, T3 1.0, FTI 51")
check("Reads lab values from text", values.get("TSH") == 14.2 and values.get("age") == 46)

# 3. Does an obviously ill panel come back as ill?
ill = client.post("/api/chat", json={"message": "46F, TSH 14.2, TT4 48, T3 1.0, FTI 51"}).json()
check("Spots an underactive thyroid", "hypothyroid" in ill["label"].lower(), ill["label"])

# 4. Does a normal panel come back normal?
well = client.post("/api/chat", json={"message": "30M TSH 1.8 TT4 110 T3 2.0 FTI 110"}).json()
check("Spots a normal thyroid", "No hypothyroidism" in well["label"], well["label"])

# 5. Does it refuse to guess from a single value?
thin = client.post("/api/chat", json={"message": "my TSH is 3.8"}).json()
check("Refuses to guess from one value", thin["type"] == "text")

# 6. Does the Q&A search work, and does it know its limits?
qa = ThyroidQA()
check("Q&A answers a real question", "TSH" in qa.answer("what is tsh")["answer"])
check("Q&A admits when it does not know", not qa.answer("capital of France")["found"])

# 7. Does the report reader convert units correctly?
from report_reader import parse_report                # noqa: E402
sample = "SERUM THYROXINE, T4 98 ng/mL 52-127\nAge / Sex : 42 YRS / M"
parsed = parse_report(sample)
check("Converts ng/mL to nmol/L", abs(parsed["values"]["TT4"] - 126.13) < 0.1,
      f"98 ng/mL -> {parsed['values'].get('TT4')} nmol/L")
check("Reads age and sex from a report header",
      parsed["values"].get("age") == 42 and parsed["values"].get("sex") == 1.0)

# 8. Does it ignore the digit inside a test code like T3?
t3 = parse_report("SERUM TRIIODOTHYRONINE, T3 1.86 ng/mL 0.69 - 2.15")
check("Does not mistake the 3 in T3 for a value",
      abs(t3["values"]["T3"] - 2.857) < 0.01, f"1.86 ng/mL -> {t3['values'].get('T3')}")

# 9. Does it flag a suspicious value instead of trusting it?
odd = parse_report("THYROID-STIMULATING HORMONE, TSH 33 uIU/mL 0.3-4.5")
check("Flags a suspicious TSH for checking", len(odd["warnings"]) > 0)

# 10. Doctor mode: does batch triage rank the sickest patient first?
batch = """P001: 30M TSH 1.8 TT4 110 T3 2.0 FTI 110
P002: 46F TSH 14.2 TT4 48 T3 1.0 FTI 51
P003: 62F TSH 9.1 FTI 70"""
tri = client.post("/api/triage", json={"text": batch}).json()
check("Batch triage assesses every valid row", tri["assessed"] == 3)
check("Sickest patient is ranked first",
      tri["worklist"][0]["label"] == "P002", tri["worklist"][0]["label"])
check("Healthy patient is ranked last",
      tri["worklist"][-1]["label"] == "P001", tri["worklist"][-1]["label"])

# 11. Does triage refuse partial rows instead of guessing?
partial = client.post("/api/triage", json={"text": "P009: TSH 3.8\nP010: 46F TSH 14.2 FTI 51"}).json()
check("Triage skips rows with too little data", len(partial["skipped"]) == 1)

# 12. Are clinical flags raised where they should be?
preg = client.post("/api/triage", json={"text": "P011: 30F TSH 62 TT4 40 FTI 45, pregnant"}).json()
flags = " ".join(preg["worklist"][0]["flags"]).lower()
check("Flags pregnancy and marked TSH elevation",
      "pregnan" in flags and "50" in flags)

# 13. Is a clinical note generated for clinicians only?
doc = client.post("/api/chat", json={"message": "46F TSH 14.2 FTI 51", "mode": "doctor"}).json()
pat = client.post("/api/chat", json={"message": "46F TSH 14.2 FTI 51", "mode": "patient"}).json()
check("Clinical note generated in doctor mode", bool(doc.get("note")))
check("No clinical note in patient mode", pat.get("note") is None)

# 14. Is the website being served?
check("Home page loads", client.get("/").status_code == 200)
check("Model switcher list loads", len(client.get("/api/models").json()["models"]) == 2)
check("Scorecard loads", client.get("/api/scorecard").json()["macro_f1"] > 0.9)

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed.")
print("Ready to demo.\n" if passed == len(results)
      else "Something is broken. Read the FAIL lines above.\n")
sys.exit(0 if passed == len(results) else 1)
