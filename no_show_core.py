"""Core logic for the appointment no-show tool.

Mirrors the Colab notebook: same cleaning rules, same features, same
patient-safe history counts, and a logistic regression model with
balanced class weights.
"""
import io

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOP_NB = 20
SHARE = 0.10  # default capacity: call the top 10% of appointments
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

NUMERIC = [
    "age", "lead_days", "weekday", "sms_received", "scholarship",
    "hypertension", "diabetes", "alcoholism", "handicap", "female",
    "prior_appts", "prior_noshows", "prior_noshow_rate",
]

RENAME = {
    "PatientId": "patient_id", "AppointmentID": "appointment_id", "Gender": "gender",
    "Age": "age", "Neighbourhood": "neighbourhood", "Scholarship": "scholarship",
    "Hipertension": "hypertension", "Diabetes": "diabetes", "Alcoholism": "alcoholism",
    "Handcap": "handicap", "SMS_received": "sms_received", "No-show": "no_show",
}

FRIENDLY = {
    "lead_days": "booked many days in advance",
    "sms_received": "reminder SMS received",
    "age": "patient age",
    "prior_noshows": "previous no-shows",
    "prior_noshow_rate": "share of past appointments missed",
    "prior_appts": "number of past appointments",
    "scholarship": "welfare scholarship",
    "weekday": "visit day of the week",
    "female": "female patient",
    "hypertension": "hypertension",
    "diabetes": "diabetes",
    "alcoholism": "alcoholism",
    "handicap": "disability",
}


def read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes), dtype={"PatientId": str, "AppointmentID": str})


def clean(raw):
    """Apply the notebook's cleaning rules. Returns (clean_df, rows_removed)."""
    d = raw.rename(columns=RENAME).copy()
    d["scheduled"] = pd.to_datetime(d["ScheduledDay"], utc=True).dt.tz_convert(None).dt.normalize()
    d["appointment"] = pd.to_datetime(d["AppointmentDay"], utc=True).dt.tz_convert(None).dt.normalize()
    d["lead_days"] = (d["appointment"] - d["scheduled"]).dt.days
    n0 = len(d)
    d = d[d["age"].between(0, 110) & (d["lead_days"] >= 0)].copy()
    removed = n0 - len(d)
    d["handicap"] = (d["handicap"] > 0).astype(int)
    d["no_show_flag"] = (d["no_show"] == "Yes").astype(int)
    d["female"] = (d["gender"] == "F").astype(int)
    d = d.sort_values(["scheduled", "appointment"]).reset_index(drop=True)
    return d, removed


def add_history(d):
    """Patient history using only earlier appointments (no leakage)."""
    d = d.copy()
    d["weekday"] = d["appointment"].dt.dayofweek
    g = d.groupby("patient_id")
    d["prior_appts"] = g.cumcount()
    d["prior_noshows"] = g["no_show_flag"].cumsum() - d["no_show_flag"]
    d["prior_noshow_rate"] = (d["prior_noshows"] / d["prior_appts"].replace(0, np.nan)).fillna(0)
    return d


def neighbourhood_groups(d, top_nb):
    return d["neighbourhood"].where(d["neighbourhood"].isin(top_nb), "Other")


def build_X(d, bundle):
    X = d[NUMERIC].astype(float).copy()
    grp = neighbourhood_groups(d, bundle["top_nb"])
    for c in bundle["nb_cats"][1:]:
        X[f"nb_{c}"] = (grp == c).astype(float)
    return X[bundle["columns"]]


def train(file_bytes):
    """Train the model on an uploaded Kaggle-format CSV. Returns (bundle, scored_df, X)."""
    raw = read_csv(file_bytes)
    d, removed = clean(raw)
    d = add_history(d)
    top_nb = d["neighbourhood"].value_counts().nlargest(TOP_NB).index.tolist()
    nb_cats = sorted(top_nb) + ["Other"]
    columns = NUMERIC + [f"nb_{c}" for c in nb_cats[1:]]
    bundle = {"top_nb": top_nb, "nb_cats": nb_cats, "nb_ref": nb_cats[0], "columns": columns,
              "rows_removed": removed}
    X = build_X(d, bundle)
    y = d["no_show_flag"]
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X, y)
    bundle["model"] = model
    scores = model.predict_proba(X)[:, 1]
    bundle["cutoff_top10"] = float(np.quantile(scores, 1 - SHARE))
    d["risk"] = scores
    return bundle, d, X


def contributions(bundle, X):
    """Per-feature contribution to the log-odds (scaled value x coefficient)."""
    scaler, lr = bundle["model"][0], bundle["model"][-1]
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled * lr.coef_[0], index=X.index, columns=X.columns)


def friendly(name):
    if name.startswith("nb_"):
        return "neighbourhood: " + name[3:]
    return FRIENDLY.get(name, name)


def top_reason(contrib_row):
    """Main factor and whether it raises or lowers risk."""
    name = contrib_row.abs().idxmax()
    direction = "raises" if contrib_row[name] > 0 else "lowers"
    return f"{friendly(name)} ({direction} risk)"


def single_row(bundle, v):
    """Build one feature row from the form values v (dict)."""
    row = {c: 0.0 for c in bundle["columns"]}
    for k in NUMERIC:
        if k in v:
            row[k] = float(v[k])
    row["prior_noshow_rate"] = v["prior_noshows"] / v["prior_appts"] if v["prior_appts"] > 0 else 0.0
    nb = v["neighbourhood"]
    if nb != bundle["nb_ref"] and f"nb_{nb}" in row:
        row[f"nb_{nb}"] = 1.0
    return pd.DataFrame([row])[bundle["columns"]]


def risk_band(p, cutoff):
    if p >= cutoff:
        return "HIGH"
    if p >= 0.5 * cutoff:
        return "MEDIUM"
    return "LOW"


def recall_by_age_band(d, flags):
    bands = pd.cut(d["age"], bins=[-1, 17, 40, 60, 110], labels=["0-17", "18-40", "41-60", "60+"])
    t = pd.DataFrame({"band": bands.astype(str), "flag": flags, "y": d["no_show_flag"].values})
    pos = t[t["y"] == 1]
    return pos.groupby("band")["flag"].mean()


def capacity_flags(scores, share=SHARE):
    k = int(len(scores) * share)
    flags = np.zeros(len(scores), dtype=bool)
    flags[np.argsort(-np.asarray(scores))[:k]] = True
    return flags
