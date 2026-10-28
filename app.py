"""Appointment No-Show Risk Tool (Streamlit).

Run:  streamlit run app.py
Upload the Kaggle file noshowappointments-kagglev2-may-2016.csv in the sidebar.
"""
import pandas as pd
import streamlit as st

import no_show_core as core

st.set_page_config(page_title="Appointment No-Show Risk Tool", layout="wide")

st.title("Appointment No-Show Risk Tool")
st.caption("Ranks appointments by no-show risk so clinic staff can prioritise reminder calls "
           "and rebooking offers. Decision support only: staff make the final call.")

# ---------- Sidebar: data and capacity ----------
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload appointment CSV", type=["csv"])
    st.header("Capacity")
    share = st.slider("Share of appointments staff can contact", 0.05, 0.30, core.SHARE, 0.05,
                      help="The top share of appointments by risk is flagged for a call.")

if uploaded is None:
    st.info("Upload noshowappointments-kagglev2-may-2016.csv in the sidebar to train the model "
            "and enable all tabs.")
    st.stop()

with st.spinner("Training the model on the uploaded data..."):
    bundle, data, X = core.train(uploaded.getvalue())
    contrib_all = core.contributions(bundle, X)
    scores = data["risk"].values

flags = core.capacity_flags(scores, share)
flagged_n = int(flags.sum())

tab_score, tab_list, tab_dash, tab_model = st.tabs(
    ["Score one appointment", "Daily call list", "Clinic dashboard", "Model and fairness"])

# ---------- Tab 1: score one appointment ----------
with tab_score:
    st.subheader("Score a single upcoming appointment")
    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 0, 110, 40)
        gender = c1.selectbox("Gender", ["F", "M"])
        neighbourhood = c1.selectbox("Neighbourhood", bundle["nb_cats"])
        prior_appts = c1.number_input("Past appointments", 0, 500, 3)
        lead = c2.number_input("Days between booking and visit", 0, 365, 14)
        weekday = c2.selectbox("Visit day", core.WEEKDAYS)
        sms = c2.checkbox("Reminder SMS sent", value=True)
        prior_ns = c2.number_input("Past no-shows", 0, 500, 0)
        scholar = c3.checkbox("Welfare scholarship")
        hyp = c3.checkbox("Hypertension")
        dia = c3.checkbox("Diabetes")
        alc = c3.checkbox("Alcoholism")
        hand = c3.checkbox("Disability")
        submitted = st.form_submit_button("Score appointment")

    if submitted:
        if prior_ns > prior_appts:
            st.error("Past no-shows cannot be more than past appointments.")
        else:
            v = {"age": age, "female": int(gender == "F"), "lead_days": lead,
                 "weekday": core.WEEKDAYS.index(weekday), "sms_received": int(sms),
                 "scholarship": int(scholar), "hypertension": int(hyp), "diabetes": int(dia),
                 "alcoholism": int(alc), "handicap": int(hand), "prior_appts": prior_appts,
                 "prior_noshows": prior_ns, "neighbourhood": neighbourhood}
            row = core.single_row(bundle, v)
            p = float(bundle["model"].predict_proba(row)[0, 1])
            band = core.risk_band(p, bundle["cutoff_top10"])

            m1, m2 = st.columns(2)
            m1.metric("Model risk score", f"{p:.1%}")
            m2.metric("Risk band", band)
            if band == "HIGH":
                st.error("HIGH: send a reminder and offer rebooking to an earlier or more "
                         "convenient slot.")
            elif band == "MEDIUM":
                st.warning("MEDIUM: send a standard reminder.")
            else:
                st.success("LOW: no extra action needed.")

            st.markdown("**Main reasons for this score**")
            ct = core.contributions(bundle, row).iloc[0]
            top = ct.reindex(ct.abs().sort_values(ascending=False).index)[:3]
            for name, val in top.items():
                direction = "raises" if val > 0 else "lowers"
                st.write(f"- {core.friendly(name)}: {direction} risk")
            st.caption("Score from a logistic regression trained on the uploaded data. The score "
                       "is a ranking aid, not a calibrated probability.")

# ---------- Tab 2: daily call list ----------
with tab_list:
    st.subheader("Ranked call list")
    st.caption("Ranks the uploaded appointments by risk. In live use, this list would be built "
               "from upcoming bookings only.")
    order = scores.argsort()[::-1][:flagged_n]
    top_idx = data.index[order]
    reason = contrib_all.loc[top_idx].apply(core.top_reason, axis=1)
    calls = pd.DataFrame({
        "Appointment ID": data.loc[top_idx, "appointment_id"].values,
        "Visit date": data.loc[top_idx, "appointment"].dt.date.values,
        "Age": data.loc[top_idx, "age"].values,
        "Lead days": data.loc[top_idx, "lead_days"].values,
        "Reminder SMS": data.loc[top_idx, "sms_received"].map({0: "No", 1: "Yes"}).values,
        "Risk score": (scores[order] * 100).round(1),
        "Main reason": reason.values,
    })
    st.write(f"Flagged for a call: {flagged_n:,} of {len(data):,} appointments "
             f"({share:.0%}).")
    st.dataframe(calls, hide_index=True)
    st.download_button("Download call list (CSV)", calls.to_csv(index=False).encode(),
                       "call_list.csv", "text/csv")

# ---------- Tab 3: dashboard ----------
with tab_dash:
    st.subheader("Clinic dashboard (uploaded data)")
    no_show = data["no_show_flag"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Appointments", f"{len(data):,}")
    k2.metric("No-show rate", f"{no_show.mean():.1%}")
    k3.metric("Attended share (utilisation proxy)", f"{1 - no_show.mean():.1%}")

    buckets = pd.cut(data["lead_days"], bins=[-1, 2, 7, 14, 30, 365],
                     labels=["0-2 days", "3-7 days", "8-14 days", "15-30 days", "31+ days"])
    st.markdown("**No-show rate by booking lead time (%)**")
    st.bar_chart(no_show.groupby(buckets, observed=True).mean().mul(100).rename(index=str))

    st.markdown("**No-show rate by reminder SMS (%)**")
    sms_rate = no_show.groupby(data["sms_received"]).mean().mul(100)
    st.bar_chart(sms_rate.rename(index={0: "No reminder", 1: "Reminder sent"}))

    bands = pd.cut(data["age"], bins=[-1, 17, 40, 60, 110], labels=["0-17", "18-40", "41-60", "60+"])
    st.markdown("**No-show rate by age band (%)**")
    st.bar_chart(no_show.groupby(bands, observed=True).mean().mul(100).rename(index=str))

# ---------- Tab 4: model and fairness ----------
with tab_model:
    st.subheader("Model comparison and fairness")
    st.markdown("**Validation results from the Colab notebook** (validation set, top 10% capacity)")
    st.table(pd.DataFrame({
        "Model": ["Benchmark (lead time)", "Logistic regression (selected)", "Random forest",
                  "XGBoost"],
        "ROC-AUC": [0.689, 0.681, 0.742, 0.743],
        "PR-AUC": [0.297, 0.330, 0.392, 0.398],
        "Recall at 10%": [0.154, 0.188, 0.222, 0.224],
        "Age recall gap": [0.015, 0.163, 0.288, 0.280],
    }))
    st.caption("Selection rules set in advance: recall at capacity must be at least the benchmark's, "
               "and the age-band recall gap must be within 5 percentage points. No trained model met "
               "the fairness rule, so logistic regression was selected by the notebook's default.")

    st.markdown("**Recall by age band for the tool's model (uploaded data, same capacity)**")
    st.caption("Shows how many of each age group's no-shows were flagged. Computed on the "
               "uploaded data, so it is an in-sample check, not the held-out test result.")
    rec = core.recall_by_age_band(data, flags)
    st.bar_chart(rec.mul(100).rename("Recall %"))
    gap = float(rec.max() - rec.min()) if len(rec) > 1 else 0.0
    if gap > 0.05:
        st.warning(f"Recall differs by {gap:.1%} between age bands, above the 5-point limit. "
                   "Review before using the list for calls.")
    else:
        st.success(f"Recall differs by {gap:.1%} between age bands, within the 5-point limit.")
    st.caption(f"Rows removed during cleaning: {bundle['rows_removed']}. Probabilities use balanced "
               "class weights, so they are not calibrated; use them for ranking only.")
