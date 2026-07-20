import streamlit as st
import pandas as pd
import pickle

MODELS_DIR = "models"

# ---- Load saved artifacts ----
with open(f'{MODELS_DIR}/final_model.pkl', 'rb') as f:
    final_model = pickle.load(f)

with open(f'{MODELS_DIR}/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(f'{MODELS_DIR}/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open(f'{MODELS_DIR}/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

with open(f'{MODELS_DIR}/numerical_cols.pkl', 'rb') as f:
    numerical_cols = pickle.load(f)

binary_cols = ['gender', 'ssc_b', 'hsc_b', 'workex', 'specialisation']

st.set_page_config(page_title="Student Placement Predictor", page_icon=":mortar_board:")
st.title("Student Placement Predictor")
st.write("Enter a student's academic and demographic details to predict placement outcome.")

# ---- Input form ----
with st.form("student_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        ssc_p = st.slider("10th Grade % (ssc_p)", 40.0, 100.0, 70.0)
        ssc_b = st.selectbox("10th Board (ssc_b)", ["Central", "Others"])
        hsc_p = st.slider("12th Grade % (hsc_p)", 40.0, 100.0, 70.0)
        hsc_b = st.selectbox("12th Board (hsc_b)", ["Central", "Others"])
        hsc_s = st.selectbox("12th Stream (hsc_s)", ["Commerce", "Science", "Arts"])

    with col2:
        degree_p = st.slider("Degree %", 40.0, 100.0, 65.0)
        degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
        workex = st.selectbox("Work Experience", ["Yes", "No"])
        etest_p = st.slider("Employability Test %", 40.0, 100.0, 70.0)
        specialisation = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"])
        mba_p = st.slider("MBA %", 40.0, 100.0, 62.0)

    submitted = st.form_submit_button("Predict Placement")

# ---- Prediction ----
if submitted:
    new_student = pd.DataFrame({
        'gender': [gender], 'ssc_p': [ssc_p], 'ssc_b': [ssc_b],
        'hsc_p': [hsc_p], 'hsc_b': [hsc_b], 'hsc_s': [hsc_s],
        'degree_p': [degree_p], 'degree_t': [degree_t], 'workex': [workex],
        'etest_p': [etest_p], 'specialisation': [specialisation], 'mba_p': [mba_p]
    })

    encoded = new_student.copy()
    for col in binary_cols:
        encoded[col] = label_encoders[col].transform(encoded[col])

    encoded = pd.get_dummies(encoded, columns=['hsc_s', 'degree_t'], drop_first=True, dtype=int)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    scaled = encoded.copy()
    scaled[numerical_cols] = scaler.transform(encoded[numerical_cols])

    prediction = final_model.predict(scaled)[0]
    probability = final_model.predict_proba(scaled)[:, 1][0]

    st.divider()

    status_color = "#1a7f37" if prediction == 1 else "#c9302c"
    status_text = "Placed" if prediction == 1 else "Not Placed"
    st.markdown(
        f'<div style="padding:16px 20px;border-radius:8px;background-color:{status_color}15;'
        f'border:1px solid {status_color}40;margin-bottom:10px;">'
        f'<span style="font-size:1.4rem;font-weight:600;color:{status_color};">'
        f'Predicted outcome: {status_text}'
        f'</span></div>',
        unsafe_allow_html=True
    )

    st.metric("Probability of Placement", f"{probability:.1%}")
    st.progress(float(probability))

    # ---- Explain WHY: per-feature contribution (scaled value x coefficient) ----
    # Only meaningful because Logistic Regression is a linear model.
    contributions = scaled.iloc[0].values * final_model.coef_[0]
    contrib_df = pd.DataFrame({
        'Feature': scaled.columns,
        'Contribution': contributions
    })
    contrib_df = contrib_df.reindex(
        contrib_df['Contribution'].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    st.subheader("Key factors behind this prediction")

    top_n = 6
    top = contrib_df.head(top_n)
    max_abs = top['Contribution'].abs().max()

    POSITIVE_COLOR = "#1a7f37"
    NEGATIVE_COLOR = "#c9302c"
    TRACK_COLOR = "#e9ecef"

    rows_html = ""
    for _, row in top.iterrows():
        pct = abs(row['Contribution']) / max_abs * 100 if max_abs > 0 else 0
        color = POSITIVE_COLOR if row['Contribution'] > 0 else NEGATIVE_COLOR
        direction = "toward Placed" if row['Contribution'] > 0 else "toward Not Placed"
        rows_html += (
            f'<div style="margin-bottom:14px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.9rem;margin-bottom:4px;">'
            f'<span style="font-weight:500;">{row["Feature"]}</span>'
            f'<span style="color:{color};">{direction}</span>'
            f'</div>'
            f'<div style="background-color:{TRACK_COLOR};border-radius:4px;height:8px;width:100%;">'
            f'<div style="background-color:{color};width:{pct}%;height:8px;border-radius:4px;"></div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(f'<div style="margin-top:8px;">{rows_html}</div>', unsafe_allow_html=True)

    st.caption(
        "This prediction is based on a Logistic Regression model trained on 215 historical "
        "student records. Results are indicative, not a guarantee of placement outcome. "
        "The factors above reflect this model's learned pattern, not a causal claim."
    )
