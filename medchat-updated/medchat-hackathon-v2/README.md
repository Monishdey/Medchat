# MedChat

A chat app that reads your thyroid blood test results and tells you whether they
point to an underactive thyroid — and answers your questions about what it all
means.

Think ChatGPT, but instead of switching between GPT-4 and GPT-5, you switch
between **disease models**. Right now there is one working model, **T1 (Thyroid)**.
A second one, **C1 (Cancer)**, sits in the menu marked "coming soon".

**No API keys. No paid services. No internet needed once it is installed.**

---

## Table of contents

1. [What this project actually does](#1-what-this-project-actually-does)
2. [How it works, in plain English](#2-how-it-works-in-plain-english)
3. [Setup, step by step](#3-setup-step-by-step)
4. [The dataset — where it came from and why](#4-the-dataset--where-it-came-from-and-why)
5. [What "training a model" actually means](#5-what-training-a-model-actually-means)
6. [The metrics — what the numbers mean](#6-the-metrics--what-the-numbers-mean)
7. [The results we got](#7-the-results-we-got)
8. [What you can type in — the inputs](#8-what-you-can-type-in--the-inputs)
9. [How the Q&A works without any AI](#9-how-the-qa-works-without-any-ai)
10. [Every file explained](#10-every-file-explained)
11. [Troubleshooting](#11-troubleshooting)
12. [Demo script for judges](#12-demo-script-for-judges)
13. [Future scope](#13-future-scope)
14. [Pitch deck content](#14-pitch-deck-content)
15. [Honest limitations](#15-honest-limitations)

---

## 1. What this project actually does

Imagine you get a blood test. The report comes back covered in things like
`TSH 14.2 mIU/L` and `FTI 51`. You have no idea what any of it means, and your
next doctor's appointment is in three weeks.

MedChat does two things about that:

**Thing one — it assesses your numbers.** Type them in, or photograph the report,
and a machine learning model trained on 3,772 real patient records tells you
which of three categories your results look like:

| Result | What it means |
|---|---|
| No hypothyroidism detected | Your numbers look normal |
| Compensated hypothyroidism | Early stage. The gland is struggling but coping |
| Primary hypothyroidism | The thyroid is underactive |

It also shows how confident it is, which values pushed it that way, and what it
had to guess because you did not supply it.

**Thing two — it answers your questions.** "What is TSH?" "Can this affect
pregnancy?" "What pill do I take?" These come from a written knowledge base of
32 thyroid topics, searched instantly.

The app decides which of these two things you wanted, automatically, based on
what you typed.

### The one-sentence version

> MedChat is a medical chat interface with a swappable disease-model menu, where
> a locally trained classifier reads your blood results and a local search engine
> answers your questions — with no external AI service involved.

---

## 2. How it works, in plain English

Here is the entire flow when you press send:

```
   You type: "46F, TSH 14.2, TT4 48, T3 1.0, FTI 51"
                      |
                      v
   [ 1 ] Look for lab values in the text using pattern matching
         Found: age=46, sex=F, TSH=14.2, TT4=48, T3=1.0, FTI=51
                      |
                      v
   [ 2 ] Do we have TSH plus at least one more test?
         |                                    |
        YES                                   NO
         |                                    |
         v                                    v
   [ 3a ] Run the T1 model            [ 3b ] Search the Q&A
          -> "Primary hypothyroidism"        knowledge base
          -> 90.9% confident                 -> best matching answer
                      |                                    |
                      +------------------+-----------------+
                                         v
                              Send the answer to the browser
```

Step 2 is the entire "brain" of the chat, and it is about ten lines of code in
`app.py`. It is called **intent routing**: working out what the user wants
before deciding how to answer.

### Why there is no ChatGPT or DeepSeek in this project

An earlier version of this used the DeepSeek API. We removed it, and the project
got better. Here is the honest reasoning:

| | With an AI chat API | With this approach |
|---|---|---|
| Cost | Needs a paid key | Free forever |
| Works offline | No | Yes |
| Can invent fake facts | Yes, and it does | Impossible |
| Speed | 2–5 seconds | About 1 millisecond |
| Breaks if wifi dies mid-demo | Yes | No |

For anything health-related, "cannot invent fake facts" is not a small feature.
Every answer MedChat gives is one a human wrote down in advance. It is honest
about what it does not know instead of making something up.

The trade-off, stated plainly: MedChat cannot handle a question nobody
anticipated. That is a real limitation, and section 13 covers how to fix it.

---

## 3. Setup, step by step

Follow these in order. This assumes you have never done any of it before.

### Step 0 — Check you have Python

Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```bash
python --version
```

You should see `Python 3.10` or higher. If it says "command not found", try
`python3 --version` instead. If neither works, install Python from
[python.org/downloads](https://www.python.org/downloads/) and tick
**"Add Python to PATH"** during installation — people miss that box constantly
and then nothing works.

> Throughout this README, wherever you see `python`, use `python3` instead if
> that is what worked for you.

### Step 1 — Get into the project folder

```bash
cd path/to/medchat
```

Check you are in the right place. This should list `app.py`, `train.py` and others:

```bash
ls        # Mac/Linux
dir       # Windows
```

### Step 2 — Install Tesseract (for reading photos of lab reports)

This one is not a Python package, so it installs differently on each system.

**Windows:** download the installer from
[github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki),
run it, and note the install path (usually `C:\Program Files\Tesseract-OCR`).

**Mac:** `brew install tesseract`
(if you do not have Homebrew, get it from [brew.sh](https://brew.sh))

**Linux:** `sudo apt install tesseract-ocr`

Check it worked:

```bash
tesseract --version
```

> **You can skip this step.** Everything else still works — typing your values
> in by hand works perfectly. You just will not be able to upload photos.

### Step 3 — Create a virtual environment

A virtual environment is a private box for this project's packages, so they do
not clash with anything else on your computer. Highly recommended.

```bash
python -m venv venv
```

Then switch into it:

```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

You will know it worked because your terminal line now starts with `(venv)`.

### Step 4 — Install the Python packages

```bash
pip install -r requirements.txt
```

This downloads about 100 MB and takes a minute or two. If it fails, try
upgrading pip first with `pip install --upgrade pip`.

### Step 5 — Train the model

```bash
python train.py
```

This reads the patient data, trains the model, and prints a full report. It
takes about 10 seconds. You will see something like:

```
Loaded 3772 patient records with 30 columns.
Dropped 7 columns that would cause data leakage.
Removed 2 records from a class too rare to learn.
Fixed 1 impossible ages.

How many patients of each type:
  negative                    3481  (92.3%)
  compensated_hypothyroid      194  (5.1%)
  primary_hypothyroid           95  (2.5%)

Training on 3016 patients, testing on 754 unseen ones.

Running 5-fold cross-validation (this trains the model 5 times)...
  Scores per fold: [0.979, 0.941, 0.973, 0.962, 0.929]
  Average: 0.9567  (spread +/- 0.0189)
```

...followed by the report card. **Read that output.** Section 5 and 6 explain
every line of it.

This creates a `model/` folder. That folder is your trained model.

### Step 6 — Check everything works

```bash
python check.py
```

You should see 10 lines all saying PASS. If any say FAIL, the message tells you
what to fix. **Run this before your demo, every time.**

### Step 7 — Start the app

```bash
python app.py
```

You will see:

```
==================================================
  MedChat is starting
  Open this in your browser:  http://localhost:8000
  Press CTRL+C to stop
==================================================
```

Open **http://localhost:8000** in your browser. That is it.

To stop the server, press `CTRL+C` in the terminal.

### Quick reference

Once installed, you only ever need these two lines:

```bash
source venv/bin/activate     # or venv\Scripts\activate on Windows
python app.py
```

---

## 4. The dataset — where it came from and why

### What we used

**Thyroid Disease Data**, available on Kaggle at
[kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data)
and originally from the UCI Machine Learning Repository. The data was collected
by the Garavan Institute in Sydney, Australia.

It is included in this repo at `data/hypothyroid.csv`, so you do not need to
download anything.

| | |
|---|---|
| Rows | 3,772 patients |
| Columns | 30 |
| Target | Diagnosis, confirmed by doctors |
| Format | CSV, missing values written as `?` |
| Licence | Public, free for research and education |

### Why we picked this one, honestly

We looked at several thyroid datasets on Kaggle before choosing. This is the
comparison, because picking a dataset is a real skill and most tutorials skip it:

| Dataset | Rows | Real patients? | Why we did or did not use it |
|---|---|---|---|
| **Thyroid Disease Data** (UCI/Garavan) | 3,772 | Yes | **Chosen.** Real blood test values, real confirmed diagnoses, enough rows to train on. It is old, but it is genuine. |
| Differentiated Thyroid Cancer Recurrence (2023) | 383 | Yes | Newest real thyroid dataset available. Predicts *cancer recurrence*, not hypothyroidism, so it does not fit T1. **We included it anyway** at `data/thyroid_cancer_recurrence.csv` — it is the intended basis for C1. |
| Thyroid Cancer Risk Dataset (2025) | 212,000 | **No — synthetic** | Very popular on Kaggle, and very tempting because of the size. But the rows are computer-generated, not real people. A model trained on it learns the generator's rules, not medicine. **Rejected.** |
| Thyroid ultrasound image sets (DDTI, AUITD) | ~3,000 images | Yes | Excellent data, but images need deep learning and a GPU. Too heavy for a hackathon weekend. Noted for future work. |

**The lesson worth taking from this:** a bigger, newer dataset is not
automatically a better one. The 212,000-row synthetic set would have produced
prettier accuracy numbers and a model that means nothing. Judges ask about this,
and "we checked whether the data was real" is a strong answer.

### The honest catch about our data

This data was collected in the 1980s at one hospital in one country. Lab
techniques and reference ranges have changed since. A model trained on it should
be treated as a coursework demonstration, not a medical instrument. Section 15
covers this in full, and section 13 covers what it would take to fix.

### The columns we actually use

Out of the 30 columns, we feed the model 21:

**Six numbers:** `age`, `TSH`, `T3`, `TT4`, `T4U`, `FTI`

**Fifteen yes/no answers:** `sex`, `on_thyroxine`, `query_on_thyroxine`,
`on_antithyroid_medication`, `sick`, `pregnant`, `thyroid_surgery`,
`I131_treatment`, `query_hypothyroid`, `query_hyperthyroid`, `lithium`,
`goitre`, `tumor`, `hypopituitary`, `psych`

**And we deliberately threw away eight columns.** This is the most important
decision in the whole project, so it gets its own section below.

### Data leakage — the trap we avoided

The dataset contains columns called `TSH_measured`, `T3_measured`, and so on.
Each says whether a doctor ordered that particular test.

That sounds harmless. It is not.

A doctor only orders a thyroid test when they *already suspect* a thyroid
problem. So `TSH_measured = true` secretly means "a trained doctor looked at this
patient and got worried". If we left that column in, the model would learn:

> "Doctor ordered the test, therefore patient is ill"

It would score beautifully on our test data and be completely useless on a new
patient, because it learned a shortcut instead of learning medicine. This is
called **data leakage**, and it is the single most common way student ML projects
quietly fail.

We dropped all seven `_measured` columns. We also dropped `TBG`, which is empty
for 96% of patients and therefore teaches nothing.

**Say this out loud to judges.** It shows you understood the data rather than
just feeding a CSV to a library.

---

## 5. What "training a model" actually means

If you have never trained a model before, here is the whole idea without any
jargon.

### The concept

You have a spreadsheet where each row is a patient. Most columns are facts about
them (age, TSH, and so on). One column is the answer (their diagnosis).

Training means: show a program thousands of these rows and let it work out for
itself which combinations of facts lead to which answer. It is not told the
rules. It finds patterns.

Afterwards you can hand it a patient it has never seen and it applies what it
found.

### The five steps, matching the code in `train.py`

**Step 1 — Clean the data.**
Computers cannot do arithmetic on the letter `t`. So `t`/`f` becomes `1`/`0`,
`M`/`F` becomes `1`/`0`, and `?` becomes "blank". We also delete impossible
values — the dataset genuinely contains someone aged 455.

**Step 2 — Split into training and testing.**
We hide 20% of patients away and never let the model see them during training.

Why? Imagine revising for an exam using the actual exam paper. You would score
100% and learn nothing. Testing a model on data it trained on is exactly that
mistake. The hidden 20% is the real exam.

We use `stratify=y`, which keeps the same proportion of ill and healthy patients
in both halves. Without it, random chance could put nearly all the rare cases in
one side and wreck your results.

**Step 3 — Fill in the blanks.**
Real medical data has gaps: not every patient got every test. `SimpleImputer`
fills each gap with the median of that column.

Crucially, this lives *inside* the pipeline. That means the same filling
happens automatically at prediction time. Forgetting this — cleaning your
training data one way and your live input another way — is the number one
hackathon bug and it is silent. Nothing crashes. The predictions just quietly go
wrong.

**Step 4 — Train the Random Forest.**

A **decision tree** is a flowchart of yes/no questions:

```
              Is TSH above 6.5?
                /          \
             yes            no
             /                \
   Is FTI below 60?         "negative"
      /        \
    yes         no
     |           |
"primary     "compensated
 hypothyroid"  hypothyroid"
```

One tree on its own is unreliable — it latches onto quirks in whichever patients
it happened to see. So a **Random Forest** builds 400 trees, each on a slightly
different random slice of the data, and lets them vote. Errors cancel out, and
the majority answer is far more robust. It is the "ask 400 people instead of one"
principle.

**One setting matters enormously here:** `class_weight="balanced"`.

92% of our patients are healthy. Without this setting, the model could answer
"healthy" to literally everything and score 92% — while catching zero sick
patients. `balanced` makes mistakes on rare classes cost proportionally more,
forcing the model to actually pay attention to the 2.5% of patients who have
primary hypothyroidism. Those are the patients the whole app exists for.

**Step 5 — Test and score.**
Run the model on the hidden 20% and compare its answers to the truth. That is
section 6.

### Cross-validation — the practice exam

Before the real test, `train.py` runs **5-fold cross-validation**. It chops the
training data into five parts, then trains five separate times, each time using a
different part as a mini-test.

You get five scores. What matters is not just the average but the **spread**:

- Five scores clustered tightly (0.97, 0.96, 0.98, 0.96, 0.97) → the model is
  stable and your final number is trustworthy.
- Five scores all over the place (0.99, 0.71, 0.94, 0.62, 0.88) → your single
  test score was luck. Do not trust it.

Ours came out at **0.9567 average with a spread of ±0.0189**. Tight. Good.

---

## 6. The metrics — what the numbers mean

This is the section judges probe hardest. Learn it properly.

### Why accuracy is a trap here

Accuracy = "what percentage did it get right".

Our data is **92.3% healthy patients**. So this useless program scores 92%:

```python
def predict(patient):
    return "negative"    # always says healthy, never looks at anything
```

92% accuracy, zero medical value, catches not a single sick patient. Accuracy is
misleading whenever your classes are unbalanced — which in medicine is nearly
always, because most people do not have any given disease.

So we barely mention accuracy. We use precision, recall and F1 instead.

### Precision and recall

Every prediction lands in one of four buckets:

|  | Model says ill | Model says healthy |
|---|---|---|
| **Actually ill** | True Positive ✓ | False Negative ✗ |
| **Actually healthy** | False Positive ✗ | True Negative ✓ |

**Precision** — when it says "ill", how often is it right?

```
Precision = True Positives / (True Positives + False Positives)
```

Low precision means false alarms. Healthy people getting frightened.

**Recall** — of all the genuinely ill people, how many did it catch?

```
Recall = True Positives / (True Positives + False Negatives)
```

Low recall means missed cases. Sick people told they are fine.

**In medical screening, recall matters more than precision.** A false alarm
costs someone an unnecessary follow-up test. A missed case costs someone months
or years of untreated illness. If you have to sacrifice one, sacrifice
precision.

### F1 score

Precision and recall pull against each other. Predict "ill" for everyone and you
get perfect recall with terrible precision. F1 combines both into one number:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

It is a harmonic mean, which is a fancy way of saying it punishes imbalance.
Precision 1.0 with recall 0.0 gives F1 = 0, not 0.5. You cannot cheat it by
maxing out one side.

### Macro F1 — the number we actually report

**Macro F1** calculates F1 separately for each of the three classes, then takes
a plain average — treating all three as equally important.

That last part is the point. Our rarest class has 95 patients out of 3,772. A
weighted average would let the huge "healthy" group drown it out. Macro F1 will
not let that happen: doing badly on the rare disease tanks the score, no matter
how well you do on healthy patients.

**This is why macro F1 is the headline metric for this project.** If someone asks
"why not just report accuracy", that is your answer.

### The confusion matrix

A grid of what got confused with what. Rows are the truth, columns are the guess.

```
                              said compensated  said negative  said primary
true compensated_hypothyroid              39              0             0
true negative                              1            694             1
true primary_hypothyroid                   0              0            19
```

Read it diagonally: 39 + 694 + 19 = 752 correct out of 754. The two mistakes are
both on the `negative` row — two healthy patients flagged as possibly ill. Those
are false alarms.

**The bottom-left region is empty, and that is the good news.** Not one ill
patient was told they were healthy. In a screening tool, that is the mistake that
actually hurts, and we made zero of them.

### Feature importance

After training, we ask the model which columns it leaned on:

```
TSH        0.4494  ##########################
FTI        0.2241  #############
TT4        0.1811  ##########
T3         0.0605  ###
T4U        0.0279  #
age        0.0235  #
```

**This is a sanity check, not a score.** TSH is the test doctors consider most
important for thyroid function, and the model independently decided the same
thing. That is strong evidence it learned real physiology rather than a quirk of
the spreadsheet.

If `age` or `referral_source` had come out on top, something would be badly
wrong and you would need to go back to the data.

---

## 7. The results we got

Everything below comes from running `python train.py` on the 754 patients the
model never saw during training.

### Headline numbers

| Metric | Score | What it means |
|---|---|---|
| **Macro F1** | **0.9868** | The headline. Strong across all three classes, including the rare ones. |
| Cross-validation F1 | 0.9567 ± 0.0189 | Stable across five different splits. Not a fluke. |
| Accuracy | 0.9973 | Reported for completeness only. See section 6 for why it is misleading. |

### Per-class breakdown

| Class | Precision | Recall | F1 | Patients |
|---|---|---|---|---|
| negative | 1.0000 | 0.9971 | 0.9986 | 696 |
| compensated_hypothyroid | 0.9750 | 1.0000 | 0.9873 | 39 |
| primary_hypothyroid | 0.9500 | 1.0000 | 0.9744 | 19 |

**Recall is 1.00 on both disease classes.** Every single ill patient in the test
set was caught. Precision is slightly lower on those classes, meaning a couple of
healthy people were flagged for a second look. For a screening tool, that is
exactly the trade-off you want.

### How to talk about this without overclaiming

If a judge says "99% accuracy, that seems too good", the right answer is not to
defend the number. It is:

> "Accuracy is inflated here because 92% of the dataset is healthy — a model that
> always guessed healthy would score 92%. That is why we report macro F1 at 0.987
> instead. The bigger caveat is the data itself: it is from one hospital in the
> 1980s, and the model has never seen a present-day patient. The score is real
> for this dataset. It is not a claim about the real world."

That answer earns more credit than the score does.

---

## 8. What you can type in — the inputs

### Method 1 — Type your values in plain language

You do not need any special format. All of these work:

```
46F, TSH 14.2, TT4 48, T3 1.0, FTI 51
TSH is 14.2 and my FTI is 51, I'm a 46 year old woman
62 M TSH 0.9 total T4 105
F/33 TSH 6.1 FTI 88
Thyroid Stimulating Hormone 12.4, Free Thyroxine Index 55
```

The pattern matcher in `app.py` handles test names, their abbreviations, their
full medical names, units, colons and dots between the name and the number.

### What each input means

| Input | Full name | Typical range | Why it matters |
|---|---|---|---|
| **TSH** | Thyroid Stimulating Hormone | 0.4–4.0 mIU/L | **Required.** The single most important value. Made by the brain to tell the thyroid to work harder, so it goes *up* when the thyroid is underactive. |
| **FTI** | Free Thyroxine Index | 60–155 | Second most important. An estimate of usable hormone. |
| **TT4** | Total Thyroxine | 60–140 nmol/L | The main hormone the thyroid produces. |
| **T3** | Triiodothyronine | 0.9–2.5 nmol/L | The more active hormone. Usually stays normal until late. |
| **T4U** | T4 Uptake | 0.7–1.3 | How much carrier protein is available. |
| **age** | | 1–120 | Risk rises with age. |
| **sex** | | M or F | Thyroid disease is 5–8× more common in women. |

Reference ranges vary between laboratories. Always read the range printed on
your own report.

### Extra context it picks up

Mention any of these in your message and the model uses them:

`levothyroxine` or `on thyroxine` · `pregnant` · `goitre` · `thyroid surgery` ·
`nodule` · `currently sick`

For example: *"TSH 8.1, FTI 62, on levothyroxine"* is read differently from the
same numbers without that phrase — because in a treated patient, a raised TSH
usually means the dose needs adjusting, not a new diagnosis.

### The minimum required

**TSH plus at least one other test.** Below that, MedChat refuses and asks for
more instead of guessing.

This refusal is deliberate, and it is worth demonstrating. Type `my TSH is 3.8`
and watch it decline to classify. A tool that guesses from insufficient data is
worse than useless in medicine.

### Method 2 — Upload a photo of your lab report

Click the paperclip and choose a photo. Then:

1. The image is converted to greyscale and enlarged if it is small
2. Tesseract reads the text off it
3. The same pattern matcher pulls out the values
4. **The values it found are shown to you before anything is sent** so you can
   spot a misread

There is a test image at `data/sample_lab_report.png`. Uploading it correctly
extracts TSH 14.2, T3 1.0, TT4 48, T4U 0.92, FTI 51 and age 46.

### Units — the thing that will silently ruin your results

**This is the most important paragraph in the README.**

Our model was trained on data measured in **nmol/L**. Most Indian labs report T3
and T4 in **ng/mL**. Those are completely different scales:

| On your report | What the model needs | Conversion |
|---|---|---|
| T4 = 98 ng/mL | 126.1 nmol/L | × 1.287 |
| T3 = 1.86 ng/mL | 2.86 nmol/L | × 1.536 |
| T4 = 7.6 µg/dL | 97.8 nmol/L | × 12.87 |
| TSH in µIU/mL | same number in mIU/L | × 1 |

Those multipliers come from the molecular weights (T4 = 776.87 g/mol, T3 =
650.98 g/mol), not from anywhere arbitrary.

Typing `T4 98` when your report says ng/mL means the model receives a number
that is 22% of the real value. It will not error. It will just be wrong.
`report_reader.py` detects the unit printed next to each value and converts
automatically, and the interface shows you "was 98 ng/mL" underneath the
converted figure so you can see what happened.

**If you type values in by hand, convert them yourself first.**

### Why the report reader works line by line

A lab report prints each test name twice: once in the results table, and again
in the explanatory notes below it ("TSH is an anterior pituitary hormone
that..."). Our first version searched the whole page as one lump of text and
matched whichever came first, which was often a paragraph rather than a result.

The reader now examines one line at a time and skips anything that reads like
prose. Two further traps it handles:

- **The digit inside a test code.** In `SERUM TRIIODOTHYRONINE, T3 1.86`, a
  naive number search returns `3` — the 3 from "T3" — not `1.86`. We skip any
  digit with a letter directly in front of it.
- **Mangled units.** Tesseract renders `µIU/mL` as `plU/mt`, `wlU/ent` and
  worse, and `FTI` as `FT|`. We match units loosely and only inside a short
  window right after the number, because the letters "ng" also appear inside the
  word "stimulating".

### The decimal point problem, and why you get to edit the values

OCR regularly reads **3.3 as 33**. That single missing dot turns a healthy
patient into a severely underactive one, and no amount of clever code can
reliably tell the two apart — a TSH of 33 is unusual but entirely possible.

So we do two things:

1. **Flag it.** Any value well outside the normal range is highlighted in amber
   with a warning to check it.
2. **Let you fix it.** Every value read off the image appears in an editable box
   *before* anything is sent to the model. Nothing is classified until you have
   looked at it.

This is not a workaround, it is the correct design. On a real report we tested,
the uncorrected TSH of 33 produced "compensated hypothyroidism", while the
corrected 3.3 produced "no hypothyroidism detected" at 99% confidence. Same
patient, opposite answers. Never let OCR output reach a model unreviewed.

### Method 3 — Just ask a question

Anything with no lab values goes to the Q&A instead:

```
What is TSH?
Can thyroid problems affect pregnancy?
What pill do I take for an underactive thyroid?
How accurate is this model?
Is my data stored anywhere?
```

---

## 8b. Clinician mode and chat history

### Chat history

The sidebar keeps your past conversations. Click one to reload it, hover to
delete it, or clear the lot from the bottom of the panel.

**Where it is stored matters.** History lives in your browser's own
localStorage — on your machine, never sent to the server, never written into
any database we control. Clear your browser data and it is gone. That keeps the
privacy promise intact while still giving you history, and it is worth saying
out loud to judges: a health app that quietly built a patient database would be
a worse project, not a better one.

Images are deliberately not stored. The browser gives uploads a temporary
address that dies on refresh, so a reloaded chat shows the message without a
broken thumbnail.

### Clinician mode

Toggle it at the bottom of the sidebar. This is not a re-skin — it is a
different tool for a different user, built on the same model.

> A patient has **one** set of results and needs them explained.
> A doctor has **fifty** and needs to know who to look at **first**.

**Batch triage.** Paste one patient per line and MedChat returns them ranked:

```
P001: 46F TSH 14.2 TT4 48 T3 1.0 FTI 51
P002: 30M TSH 1.8 TT4 110 T3 2.0 FTI 110
Bed 7 - 62F TSH 9.1 FTI 70
P004: 55F TSH 62 TT4 40 FTI 45, pregnant
```

An identifier before a colon or dash becomes the row label. Up to 100 rows.

**The sort order is the interesting part.** Rows sort by priority band first,
then by number of clinical flags, and only then by model confidence. That
ordering is deliberate: a 98%-confident healthy patient must never outrank a
60%-confident possible hypothyroid one. Confidence is the tiebreaker, never
the criterion.

In the example above, P004 lands above P001 despite *lower* confidence, because
it raised two flags — TSH over 50, and pregnancy.

**Clinical flags** are things a clinician would want surfaced beyond the
classification: TSH ≥ 50, TSH ≥ 10 where treatment is usually considered,
suppressed TSH (outside T1's scope entirely), pregnancy, thyroxine already
prescribed, acute illness, sparse input data, and low model confidence.

**Clinical notes.** Click any row to expand it and copy a plain-text block
straight into patient records — inputs, all three probabilities, what was
missing, the flags raised, and the model's provenance and limitations. Anyone
reading those notes later can see exactly where the number came from.

**What clinician mode deliberately does not do:** it never says a patient can
be skipped. It sorts a queue; the doctor still opens every case. A tool that
told a clinician to ignore someone would be dangerous. One that says "start
here" is not.

## 9. How the Q&A works without any AI

The file is `qa.py`. Here is the whole technique.

### The problem

Someone asks *"what pill do I take for an underactive thyroid"*. Our knowledge
base has an entry titled *"How is hypothyroidism treated?"*.

Those two sentences share almost no words. Simple keyword search fails. We need
to match on **meaning**.

### The solution: TF-IDF plus cosine similarity

**TF-IDF** turns every sentence into a list of numbers. The clever part is how it
weights words:

- A word that appears in *every* entry (like "thyroid") gets a **low** score,
  because it cannot help you tell entries apart.
- A word that appears in *one* entry (like "levothyroxine") gets a **high**
  score, because it is highly identifying.

TF-IDF stands for Term Frequency × Inverse Document Frequency. "Inverse document
frequency" is just that idea: rarer word, higher weight.

**Cosine similarity** then measures the angle between two of those number lists.
Same meaning gives 1.0, nothing in common gives 0.0.

### Our specific tweaks

Three things that made a real difference:

**1. We search titles and answers separately, then blend.**
Answers are long, questions are short. Mixed into one blob, the long answers
drown out the questions. We keep them apart and score `0.7 × title + 0.3 × body`,
so a hit on the question line counts more than a hit buried in a paragraph.

**2. Every entry has tags.**
The treatment entry is tagged `pill`, `medicine`, `tablet`, `dose`. This bridges
casual words to medical ones, so "pill" finds "levothyroxine".

**3. There is a confidence floor.**
If the best match scores below `0.08`, we return "I don't know" rather than the
closest bad guess. Ask it the capital of France and it correctly refuses.

### Testing it

```bash
python qa.py
```

This runs sample questions and shows what matched. The knowledge base is plain
JSON at `data/faq.json` — open it and add your own entries. Each needs a `q`, an
`a`, and some `tags`. No retraining required; just restart the app.

### The honest trade-off

This approach **cannot answer a question nobody wrote an entry for**. That is a
genuine limitation.

In exchange, it **cannot invent a fake medical fact**, which language models do
regularly and confidently. For a health app built by students in a weekend, that
trade is the right way round. Section 13 covers how to add a language model later
without losing this safety property.

---

## 10. Every file explained

```
medchat/
├── README.md            This file
├── requirements.txt     The list of Python packages needed
│
├── train.py             Trains the T1 model. Run once. Heavily commented.
├── app.py               The web server AND the chat logic. The main file.
├── qa.py                The question-answering search engine.
├── report_reader.py     Reads lab reports: line parsing, unit conversion.
├── doctor.py            Clinician mode: batch triage, clinical notes.
├── check.py             Self-test. Run before demoing.
│
├── data/
│   ├── hypothyroid.csv                 3,772 patient records (training data)
│   ├── faq.json                        32 written thyroid Q&A entries
│   ├── thyroid_cancer_recurrence.csv   2023 dataset, for building C1 later
│   └── sample_lab_report.png           Test image for the upload feature
│
├── model/               Created by train.py — this is your trained model
│   ├── t1_model.joblib
│   └── t1_scorecard.json
│
└── static/              The website
    ├── index.html
    ├── style.css
    └── app.js
```

Seven files you might edit. That is the whole project.

### If you only read two files

**`train.py`** — everything about machine learning is in here, commented line by
line.

**`app.py`** — the `chat()` function near the bottom is the intent router. It is
about ten lines and it is the core idea of the entire app.

---

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'fastapi'`**
Your virtual environment is not active, or step 4 did not finish. Activate it
(`source venv/bin/activate`) and re-run `pip install -r requirements.txt`.

**`FileNotFoundError: model/t1_model.joblib`**
You skipped step 5. Run `python train.py`.

**`Address already in use` / `port 8000`**
Something else is on that port, often a previous MedChat you forgot to stop.
Either close it, or edit the last line of `app.py` and change `port=8000` to
`port=8001`.

**Image upload says "Image reading needs Tesseract"**
Step 2 was skipped or Tesseract is not on your PATH. On Windows, add
`C:\Program Files\Tesseract-OCR` to your PATH environment variable and restart
your terminal. Typing values manually works regardless.

**Upload works but finds no values**
Photograph the report straight on, in good light, filling the frame. Blurry or
angled photos defeat OCR. Compare against `data/sample_lab_report.png`, which is
known to work.

**The page loads but nothing happens when I send**
Check the terminal running `app.py` for a red error, and your browser console
(F12). The most common cause is that `app.py` crashed and you did not notice.

**`python` is not recognised (Windows)**
Use `py` instead of `python`, or reinstall Python with "Add to PATH" ticked.

**Everything is broken and I do not know why**
```bash
python check.py
```
It tells you exactly which part failed.

---

## 12. Demo script for judges

You have about three minutes. This order works.

**1. Open on the empty state (10s).**
"This is MedChat. It reads thyroid blood results and explains them."

**2. Click the model switcher (15s).**
Show T1 available and C1 greyed out as "coming soon". "The architecture takes
disease models as plug-ins. Adding cancer means adding a file, not rebuilding
the app."

**3. Click the `T1 · F1 0.9868` chip in the header (30s).**
The scorecard opens. Point at TSH sitting on top of the feature list. "The model
independently decided TSH matters most, which is what a doctor would say. That
tells us it learned physiology, not a quirk of the spreadsheet."

**4. Send the first starter prompt (30s).**
Prediction card appears with probability bars and reasons.

**5. Upload `data/sample_lab_report.png` (30s).**
Point at the values it pulled off the image *before* sending. "It shows you what
it read so you can catch a misread."

**6. Type `my TSH is 3.8` (20s).**
It refuses and asks for more. **Pause here.** "That is deliberate. It will not
classify without TSH plus one more test. A medical tool that guesses from
insufficient data is worse than no tool."

**7. Ask `what is TSH?` (20s).**
"No language model involved. That answer is from a written knowledge base
searched with TF-IDF. It cannot hallucinate a medical fact, and it runs offline."

**8. Close (15s).**
"No API keys, no running costs, works with the wifi off. Macro F1 of 0.987 —
macro, not accuracy, because 92% of the dataset is healthy and accuracy would
flatter us."

### Questions you will be asked

**"99% accuracy sounds too good."**
Correct instinct. Accuracy is inflated by class imbalance. Macro F1 is 0.987 and
that is the number we stand behind. The real caveat is the data's age, not the
score.

**"Is this safe to use?"**
No, and we say so on every screen. It is a screening demo on 1980s research data.
It has never seen a real present-day patient.

**"Why no ChatGPT?"**
Three reasons: it cannot hallucinate a medical fact, it costs nothing to run, and
it works offline. For health, not hallucinating is a feature worth more than
fluency.

**"What is data leakage and did you handle it?"**
Yes — see section 4. The `_measured` columns encode the doctor's suspicion. We
dropped all seven.

**"How would you make this real?"**
Section 13, points 1 and 6: modern multi-site data, and prospective validation
against real clinician decisions.

---

## 13. Future scope

Ordered by what we would build next, not by ambition.

### Near term — a weekend each

**0. Persist history server-side, but only if you add accounts properly.**
Right now history is per-browser. Moving it to a database means real patient
data at rest, which means authentication, encryption and a retention policy —
not a weekend job, and not something to bolt on casually. The current design is
the honest one for a hackathon.

**1. Ship C1, the cancer model.**
The dataset is already in this repo at `data/thyroid_cancer_recurrence.csv` — 383
real patients from a 2023 study, predicting whether differentiated thyroid cancer
comes back. It is all categorical inputs (age, gender, smoking, radiotherapy
history, tumour staging, treatment response), so the same Random Forest approach
works. Copy `train.py`, change the columns, flip `ready` to `true` in `app.py`.
Two live models in the switcher tells a far stronger story than one.

**2. Explain each individual prediction with SHAP.**
Right now we explain using fixed reference ranges. SHAP would show exactly how
much *this specific patient's* TSH pushed the result, for their case rather than
in general. `pip install shap` and about thirty lines.

**3. Save conversations.**
Add SQLite so a user can see their results over time. A graph of TSH across four
tests is more clinically meaningful than any single reading.

**4. Multi-language support.**
The knowledge base is plain JSON. Translating it into Hindi, Assamese or Bengali
is a data task, not an engineering one — and it dramatically widens who this is
actually useful for.

### Medium term — a term project

**5. Add a small local language model for rephrasing only.**
Run something like Phi-3 or Llama 3.2 locally with Ollama, but constrain it hard:
it may only rephrase entries retrieved from the knowledge base, never generate
medical facts freely. This is called RAG (retrieval-augmented generation), and it
keeps the no-hallucination property while making answers feel conversational.

**6. Retrain on modern, multi-site data.**
The single biggest weakness is the 1980s dataset. Partnering with a hospital for
even a few thousand contemporary records would change this from a demo into
something arguably testable.

**7. Handle thyroid ultrasound images.**
Datasets like DDTI and AUITD contain labelled nodule scans. A convolutional
network could classify nodules, which would make image upload do real diagnostic
work instead of only OCR. Needs a GPU.

**8. Calibrate the confidence numbers.**
When the model says 90% confident, is it right 90% of the time? Probably not
exactly. `CalibratedClassifierCV` fixes this, and honest confidence matters more
in medicine than in most domains.

### Long term — the actual product

**9. A clinician-facing version.**
Different audience, different tool: batch upload, flagging of patients needing
follow-up, integration with hospital record systems.

**10. Prospective validation.**
Run it alongside real doctors on real incoming patients and measure where they
disagree. This is the step that separates a project from a medical device, and
it requires ethics approval, a study protocol, and regulatory work.

**11. More disease models.**
The switcher was built for this. Diabetes from HbA1c panels, anaemia from a full
blood count, kidney function from creatinine — all follow the same pattern:
routine bloods that people receive without explanation.

---

## 14. Pitch deck content

Eleven slides. Copy the text, add your own visuals.

---

**Slide 1 — Title**

> # MedChat
> ### Understand your blood test in 30 seconds
> A chat assistant that reads thyroid results and explains what they mean
>
> [team names] · [hackathon name]

---

**Slide 2 — The problem**

> ## 300 million people have thyroid disorders. Up to 60% do not know it.
>
> Hypothyroidism is common, cheap to test for, and completely treatable with a
> daily tablet.
>
> But the blood report says `TSH 14.2 mIU/L` and nothing else. Patients cannot
> read it. Appointments are weeks away. Symptoms — tiredness, weight gain, low
> mood — get blamed on stress for years.
>
> **The test is not the bottleneck. Understanding the result is.**

*(Check current figures from the American Thyroid Association or WHO before
presenting.)*

---

**Slide 3 — The solution**

> ## Paste your numbers. Get an answer you can actually read.
>
> **1. Assess** — a model trained on 3,772 patient records classifies your panel
>
> **2. Explain** — which values are off, and by how much
>
> **3. Answer** — 32 thyroid topics, searched instantly
>
> Type it, or photograph the report.

*(Screenshot of the prediction card here.)*

---

**Slide 4 — Demo**

> ## Live demo
>
> 1. Paste a lab panel → prediction in under a second
> 2. Upload a photo → values read off the page automatically
> 3. Ask "what is TSH?" → answered from the knowledge base
> 4. Type one borderline value → **it refuses to guess**

---

**Slide 5 — How it works**

> ## Three parts, no external AI
>
> **Pattern matching** pulls lab values out of ordinary sentences
>
> **Random Forest** — 400 decision trees vote on the diagnosis
>
> **TF-IDF search** finds the closest written answer to your question
>
> Runs entirely on one machine. No API keys, no per-query cost, works offline.

---

**Slide 6 — The data**

> ## 3,772 real patients
>
> UCI / Garavan Institute Thyroid Disease dataset, via Kaggle
>
> **We rejected a 212,000-row alternative** because it was synthetic. Bigger is
> not better if the patients are not real.
>
> **We deleted 7 columns to prevent leakage.** They recorded whether a doctor
> ordered each test — which secretly encodes the doctor's own suspicion. Leaving
> them in would have inflated our score and taught the model nothing.

---

**Slide 7 — Results**

> ## Macro F1: 0.9868
>
> | Metric | Score |
> |---|---|
> | Macro F1 (headline) | 0.9868 |
> | Cross-validation | 0.9567 ± 0.0189 |
> | Recall, primary hypothyroidism | 1.00 |
> | Recall, compensated hypothyroidism | 1.00 |
>
> **Every ill patient in the held-out test set was caught.**
>
> We report macro F1, not accuracy — 92% of the dataset is healthy, so accuracy
> would flatter any model.

---

**Slide 8 — Why we trust it**

> ## The model learned medicine, not shortcuts
>
> Most important features, as the model ranked them:
>
> **TSH 45% · FTI 22% · TT4 18%**
>
> That is the same order a doctor would give. We never told it that — it worked
> it out from the data.
>
> Sanity checks like this catch broken models that good scores hide.

---

**Slide 9 — What it will not do**

> ## Built to know its limits
>
> - **Refuses to classify** without TSH plus one more test
> - **Cannot hallucinate** — no language model, every answer human-written
> - **Shows its uncertainty** — probabilities and missing inputs, always visible
> - **Says it is not a diagnosis** on every screen
>
> Training data is from the 1980s. This is a screening demo, not a medical device.
>
> **We would rather under-claim than mislead a patient.**

---

**Slide 10 — What is next**

> ## Roadmap
>
> **Next weekend** — ship C1 for cancer recurrence (dataset already in the repo)
>
> **Next month** — per-patient SHAP explanations · result history · Hindi and
> Assamese
>
> **Next term** — local language model for rephrasing only · retrain on modern
> hospital data · ultrasound image support
>
> **The switcher is the product.** Diabetes, anaemia and kidney function all
> follow the same pattern.

---

**Slide 11 — Close**

> # Millions of people receive blood results they cannot read.
>
> ## MedChat is where they can start.
>
> Free · offline · open source · honest about what it does not know
>
> [github link] · [team contact]

---

### Delivery notes

- **Demo before slides if you can.** A working thing beats a description of one.
- **Lead with the refusal.** Everyone shows what their AI can do. Showing what
  yours refuses to do is memorable and it signals judgement.
- **Never say "99% accurate."** Say "macro F1 of 0.987, and here is why accuracy
  would mislead you." Judges remember the team that explained the metric.
- **Say the data is old before they find it.** Volunteering your weakness reads
  as competence. Being caught hiding it reads as the opposite.

---

## 15. Honest limitations

Read this before you claim anything about MedChat.

**The data is from the 1980s.** One hospital, one country, one era of lab
equipment. Reference ranges and assay methods have moved on. A model trained on
it reflects that time and place.

**It has never seen a real present-day patient.** Every number in section 7 comes
from a held-out slice of the same old dataset. That measures whether the model
learned the dataset. It does not measure whether it would help anyone today.

**It only detects hypothyroidism.** Not hyperthyroidism, not thyroid cancer, not
nodules, not anything else. A normal result here rules out very little.

**It sees only what you type.** No history, no examination, no antibody tests, no
imaging, no medication list beyond what you mention. Doctors use all of that.

**OCR is unreliable and you must check what it read.** Photographed reports lose
decimal points, and a lost decimal point changes the answer completely. The app
shows you every extracted value for editing before it classifies anything. Use
that, every time.

**The confidence numbers are not calibrated.** "90% confident" is the model's raw
vote share, not a validated probability that it is right 90% of the time.

**The Q&A can only answer what was written down.** 32 entries, human-written. Ask
something outside them and it will say it does not know — which is the correct
behaviour, but it is still a limit.

**It is not a medical device.** Not registered, not approved, not validated, not
reviewed by any clinician. It is a student project built to learn from.

**If you are worried about your thyroid, see a doctor.** This app exists to help
you understand a piece of paper, not to replace the person trained to interpret
it.

---

## Credits and licence

**Data:** Thyroid Disease dataset, Garavan Institute Sydney, via the UCI Machine
Learning Repository and Kaggle. Thyroid cancer recurrence dataset (2023) via UCI
and Kaggle.

**Medical content:** written in plain language with reference to public patient
education material from NIDDK (nih.gov) and MedlinePlus (medlineplus.gov), both
US government publications in the public domain, and the American Thyroid
Association's patient guides.

**Built with:** scikit-learn, FastAPI, pandas, Tesseract OCR.

**Licence:** MIT for the code. The datasets keep their original licences.

Built for a hackathon. Learn from it, fork it, improve it — but please do not
deploy it at patients.
