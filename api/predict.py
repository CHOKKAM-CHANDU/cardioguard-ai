import random, io, base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle, os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# ─── Load Artifacts ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load(name): return pickle.load(open(os.path.join(BASE_DIR, name), "rb"))

model           = _load("knn_model.pkl")
scaler          = _load("scaler.pkl")
feature_names   = _load("features.pkl")
X_train_scaled  = _load("X_train_scaled.pkl")
X_train_summary = _load("X_train_summary.pkl")   # shap.kmeans result

X_train_np = X_train_scaled.values if hasattr(X_train_scaled,"values") else np.array(X_train_scaled)
print("✅ Features:", feature_names)

# ─── SHAP explainer  (using model.predict exactly like the Colab reference) ──
import shap
print("⏳ Building SHAP KernelExplainer...")
shap_explainer = shap.KernelExplainer(model.predict, X_train_summary)
print("✅ SHAP ready.")

# ─── LIME explainer ───────────────────────────────────────────────────────────
import lime, lime.lime_tabular
print("⏳ Building LIME explainer...")
np.random.seed(42)
lime_sample = X_train_np[np.random.choice(len(X_train_np), size=min(500, len(X_train_np)), replace=False)]
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data = lime_sample,
    feature_names = feature_names,
    class_names   = ["No CHD", "CHD"],
    mode          = "classification",
    random_state  = 42,
)
print("✅ LIME ready. Server starting...")

# ─── Helper: fig → base64 PNG ──────────────────────────────────────────────────
def fig_to_b64(fig=None) -> str:
    buf = io.BytesIO()
    if fig is None:
        fig = plt.gcf()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor="#0f1623", edgecolor="none")
    plt.close("all")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="CHD Risk Predictor", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ─── Schema ───────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    age: float;           sex: int
    cigsPerDay: float;    BPMeds: int
    prevalentStroke: int; prevalentHyp: int; diabetes: int
    totChol: float;       BMI: float
    heartRate: float;     glucose: float;    pulse_pressure: float
    education: int        # 1-4

# ─── Build input (matching features.pkl column names exactly) ─────────────────
def build_input(data: PatientInput):
    edu = {
        1: {"education_1.0":1,"education_2.0":0,"education_3.0":0,"education_4.0":0},
        2: {"education_1.0":0,"education_2.0":1,"education_3.0":0,"education_4.0":0},
        3: {"education_1.0":0,"education_2.0":0,"education_3.0":1,"education_4.0":0},
        4: {"education_1.0":0,"education_2.0":0,"education_3.0":0,"education_4.0":1},
    }.get(data.education, {"education_1.0":1,"education_2.0":0,"education_3.0":0,"education_4.0":0})

    raw = {"age":data.age,"sex":data.sex,"cigsPerDay":data.cigsPerDay,
           "BPMeds":data.BPMeds,"prevalentStroke":data.prevalentStroke,
           "prevalentHyp":data.prevalentHyp,"diabetes":data.diabetes,
           "totChol":data.totChol,"BMI":data.BMI,"heartRate":data.heartRate,
           "glucose":data.glucose,"pulse_pressure":data.pulse_pressure, **edu}
    df = pd.DataFrame([raw])
    for col in feature_names:
        if col not in df.columns: df[col] = 0
    df     = df[feature_names]
    scaled = scaler.transform(df)
    return df, scaled

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def serve(): return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ─── Serve all frontend pages ─────────────────────────────────────────────────
_pages = ["index","assess","validate","processing","dashboard",
          "alerts","explain","factors","action","simulate","report"]

def _page_route(name):
    def _handler():
        return FileResponse(os.path.join(FRONTEND_DIR, f"{name}.html"))
    _handler.__name__ = f"serve_{name}"
    return _handler

for _p in _pages:
    app.get(f"/{_p}")(         _page_route(_p))
    app.get(f"/{_p}.html")(    _page_route(_p))

@app.post("/predict")
def predict(data: PatientInput):
    _, scaled = build_input(data)
    pred      = int(model.predict(scaled)[0])
    prob      = float(model.predict_proba(scaled)[0][1])
    return {"prediction":pred, "probability":round(prob,4),
            "risk_label":"HIGH RISK" if pred else "LOW RISK",
            "message":("Patient HAS a 10-year CHD risk."
                       if pred else "Patient does NOT have a 10-year CHD risk.")}

@app.post("/explain")
def explain(data: PatientInput):
    user_df, scaled = build_input(data)
    plt.close("all")

    # Set dark theme for SHAP/LIME figures
    plt.rcParams.update({
        "figure.facecolor": "#0f1623",
        "axes.facecolor": "#0f1623",
        "savefig.facecolor": "#0f1623",
        "text.color": "#cbd5e1",
        "axes.labelcolor": "#cbd5e1",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "axes.edgecolor": "#2d3f60"
    })

    # ── SHAP ──────────────────────────────────────────────────────────────────
    np.random.seed(42); random.seed(42)
    shap_values = shap_explainer.shap_values(scaled, nsamples=100)

    # Force plot
    shap.force_plot(
        shap_explainer.expected_value,
        shap_values[0],
        user_df,
        matplotlib=True,
        show=False,
    )
    plt.gcf().set_facecolor("#0f1623")
    force_img = fig_to_b64()

    # Bar plot (using bar_plot for older format or manual axes)
    plt.figure(figsize=(8, 5))
    shap.bar_plot(shap_values[0], feature_names=feature_names, max_display=10, show=False)
    ax = plt.gca()
    ax.set_facecolor("#0f1623")
    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#ffffff")
    bar_img = fig_to_b64()

    # ── LIME ──────────────────────────────────────────────────────────────────
    np.random.seed(42); random.seed(42)
    exp = lime_explainer.explain_instance(
        data_row   = scaled[0],
        predict_fn = model.predict_proba,
        num_features = 10,
        num_samples  = 3000,
    )
    
    lime_fig = exp.as_pyplot_figure()
    lime_fig.set_facecolor("#0f1623")
    ax = lime_fig.gca()
    ax.set_facecolor("#0f1623")
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.title.set_color("#ffffff")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3f60")
    
    lime_img = fig_to_b64(lime_fig)

    return {
        "shap_force": force_img,   # base64 PNG
        "shap_bar":   bar_img,     # base64 PNG
        "lime":       lime_img,    # base64 PNG
    }

@app.get("/health")
def health(): return {"status":"ok","version":"3.0.0"}

# ═══════════════════════════════════════════════════════════════════════════════
# DiCE — Counterfactual Explanations
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.base import BaseEstimator, ClassifierMixin
import dice_ml

# Wrapper so DiCE can call predict on raw (unscaled) input
class ScaledKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, knn, scaler_obj, feat_names):
        self.knn        = knn
        self.scaler_obj = scaler_obj
        self.feature_names = feat_names
    def fit(self, X, y=None): return self
    def predict_proba(self, X):
        df_X = pd.DataFrame(X, columns=self.feature_names)
        return self.knn.predict_proba(self.scaler_obj.transform(df_X))
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

scaled_knn = ScaledKNN(model, scaler, feature_names)

# Reconstruct training DataFrame from KNN's internal storage + y labels
print("⏳ Building DiCE explainer...")
y_train = model._y   # stored internally by sklearn KNN
X_train_unscaled = scaler.inverse_transform(X_train_np)
train_df = pd.DataFrame(X_train_unscaled, columns=feature_names)
train_df["TenYearCHD"] = y_train.astype(int)

# Binary / categorical columns
_binary_cols = ["sex","BPMeds","prevalentStroke","prevalentHyp","diabetes",
                "education_1.0","education_2.0","education_3.0","education_4.0"]
for c in _binary_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].round().astype(int)

_continuous = ["age","cigsPerDay","totChol","BMI","heartRate","glucose","pulse_pressure"]

dice_data  = dice_ml.Data(dataframe=train_df, continuous_features=_continuous,
                          outcome_name="TenYearCHD")
dice_model = dice_ml.Model(model=scaled_knn, backend="sklearn")
dice_exp   = dice_ml.Dice(dice_data, dice_model, method="random")
print("✅ DiCE ready.")

_recommendations = {
    "cigsPerDay":    "Reduce smoking; complete cessation is strongly recommended.",
    "totChol":       "Adopt a low-saturated-fat, high-fiber diet.",
    "BMI":           "Increase physical activity and control calorie intake.",
    "glucose":       "Reduce refined sugars and improve glycemic control.",
    "heartRate":     "Engage in regular aerobic exercise and stress management.",
    "BPMeds":        "Strictly adhere to prescribed blood pressure medication.",
    "pulse_pressure":"Reduce sodium intake and improve vascular health.",
}
_skip = {"age","sex","education_1.0","education_2.0","education_3.0","education_4.0"}

@app.post("/counterfactual")
def counterfactual(data: PatientInput):
    user_df, scaled = build_input(data)
    
    # 1. Check if already Low Risk
    initial_pred = int(model.predict(scaled)[0])
    if initial_pred == 0:
        return {"is_low_risk": True, "pathways": []}

    np.random.seed(42)

    # 2. Define Clinical Constraints (Pure human logic)
    permitted_range = {}
    
    # -- Continuous: only decrease to a SAFE human limit
    _safe_mins = {
        "heartRate": 60,   # Avoid bradycardia
        "BMI": 18.5,       # Avoid underweight
        "glucose": 70,      # Avoid hypoglycemia
        "totChol": 150,     # Avoid extreme low chol
        "pulse_pressure": 35 # Avoid low pulse pressure symptoms
    }
    
    for col in _continuous:
        val = float(user_df[col].values[0])
        if col == "age":
            permitted_range[col] = [val, val]
        else:
            # Use the higher of (Train Min) or (Clinical Safe Min)
            data_min = float(train_df[col].min())
            safe_min = _safe_mins.get(col, data_min)
            permitted_range[col] = [max(data_min, safe_min), val]

    # -- Lock Smoking if they don't smoke
    if float(user_df["cigsPerDay"].values[0]) == 0:
        permitted_range["cigsPerDay"] = [0, 0]

    # -- Binary: NEVER suggest getting a disease (0 -> 1 is forbidden)
    _binary_medical = ["diabetes", "prevalentStroke", "prevalentHyp", "BPMeds"]
    for col in _binary_medical:
        val = int(user_df[col].values[0])
        if val == 0:
            permitted_range[col] = [0, 0] # Lock at 0
        else:
            permitted_range[col] = [0, 1] # allow change to 0

    # -- Lock Demographics & Education
    _fixed = ["sex", "education_1.0", "education_2.0", "education_3.0", "education_4.0"]
    for col in _fixed:
        val = int(user_df[col].values[0])
        permitted_range[col] = [val, val]

    # 3. IDENTIFY MODIFIABLE FEATURES ONLY
    # If they are 0 for a condition (smoking/diabetes), don't even let DiCE "think" about changing them.
    to_vary = []
    for col in feature_names:
        if col in _skip or col == "age": continue
        
        val = float(user_df[col].values[0])
        
        # If they don't smoke or don't have a disease, lock it out of the search entirely
        _binary_only_0_to_1 = ["diabetes", "prevalentStroke", "prevalentHyp", "BPMeds", "cigsPerDay"]
        if col in _binary_only_0_to_1 and val == 0:
            continue # Don't vary this!
            
        to_vary.append(col)

    # Identifiers to HIDE (Demographics and Disease development)
    # We now skip 'BPMeds' and 'prevalentHyp' here because we want to see their IMPROVEMENTS
    _hide_features = {"age", "sex", "education", "cigsPerDay", "diabetes", 
                      "education_1.0", "education_2.0", "education_3.0", "education_4.0",
                      "prevalentStroke"}
    
    _feature_labels = {
        "prevalentHyp": "Hypertension Control",
        "BPMeds":       "Blood Pressure Medication",
        "totChol":      "Total Cholesterol",
        "BMI":          "Body Mass Index (BMI)",
        "glucose":      "Blood Glucose Level",
        "pulse_pressure":"Pulse Pressure (BP Gap)"
    }

    try:
        cf = dice_exp.generate_counterfactuals(
            user_df, total_CFs=3, desired_class=0,
            random_seed=42
        )
    except Exception:
        return {"error": "Could not calculate clinical pathways for this profile."}

    cf_df = cf.cf_examples_list[0].final_cfs_df
    orig_row = user_df.iloc[0]
    pathways = []

    for i, row in cf_df.iterrows():
        changes = []
        for col in feature_names:
            if any(h in col for h in _hide_features): continue
                
            orig_val, new_val = float(orig_row[col]), float(row[col])
            
            # Logic for Clinical IMPROVEMENTS:
            is_improvement = False
            
            # A. Normal Factors: Only show if they DECREASE (BMI, Chol, Glucose, etc.)
            if col not in ["BPMeds"] and (orig_val - new_val) > 0.01:
                is_improvement = True
                
            # B. BP Medication: Starting meds is a win! (0 -> 1)
            if col == "BPMeds" and orig_val == 0 and new_val == 1:
                is_improvement = True
            
            # C. BP Medication: Medication remission is also a win (1 -> 0)
            if col == "BPMeds" and orig_val == 1 and new_val == 0:
                is_improvement = True

            if is_improvement:
                changes.append({
                    "feature": _feature_labels.get(col, col),
                    "from": "No" if orig_val == 0 else ("Yes" if orig_val == 1 else round(orig_val, 1)), 
                    "to":   "No" if new_val == 0 else ("Yes" if new_val == 1 else round(new_val, 1)),
                    "advice": _recommendations.get(col, "")
                })
        
        if len(changes) > 0:
            pathways.append({
                "id": len(pathways) + 1,
                "changes": changes,
                "new_prob": round(float(scaled_knn.predict_proba(row[feature_names].values.reshape(1,-1))[0][1]), 4)
            })

    if not pathways:
        return {"error": "Focus on weight management and blood pressure control."}

    return {"is_low_risk": False, "pathways": pathways}


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Progression Simulation  (20-year outlook)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/simulate")
def simulate(data: PatientInput):
    user_df, _ = build_input(data)
    start_age  = int(user_df["age"].values[0])
    np.random.seed(42)

    def _prob_smooth(df):
        base = df.values[0].copy().astype(float)
        probs = []
        for _ in range(60):
            noise = np.random.normal(0, 0.04, size=base.shape)
            row = pd.DataFrame([base + noise], columns=feature_names)
            s = scaler.transform(row)
            probs.append(float(model.predict_proba(s)[0][1]))
        return round(np.mean(probs) * 100, 2)

    results = []
    # ── Calculate Trajectories ──────────────────────────────────────────────
    for yr in range(0, 21):
        worst = user_df.copy(); best = user_df.copy(); same = user_df.copy()

        worst["age"] = start_age + yr; best["age"] = start_age + yr; same["age"] = start_age + yr

        # 1. WORST: Factors rise (innovation: simulating future trajectory)
        worst["totChol"] += yr * 4.5; worst["BMI"] += yr * 0.45; worst["glucose"] += yr * 2.5
        worst["cigsPerDay"] += yr * 1.0; worst["pulse_pressure"] += yr * 1.5

        # 2. BEST: Factors improve
        best["cigsPerDay"] = max(0, float(user_df["cigsPerDay"].values[0]) - yr * 2)
        best["totChol"]    = max(160, float(user_df["totChol"].values[0]) - yr * 3)
        best["BMI"]        = max(20, float(user_df["BMI"].values[0]) - yr * 0.25)

        # 3. SAME: Only age increases (Baseline)
        # (same df is just user_df with higher age)

        results.append({
            "age":   start_age + yr,
            "worst": _prob_smooth(worst),
            "best":  _prob_smooth(best),
            "same":  _prob_smooth(same)
        })

    # Detect "Danger Age" (Innovation: Threshold crossing alert)
    danger_age = None
    for r in results:
        if r["worst"] >= 50:
            danger_age = r["age"]
            break

    # Get specific stats for the 10-year mark (Innovation focus)
    idx_10 = 10
    summary = {
        "current":  results[0]["same"],
        "baseline": results[idx_10]["same"],
        "unhealthy":results[idx_10]["worst"],
        "healthy":  results[idx_10]["best"],
        "danger_age": danger_age
    }

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1220"); ax.set_facecolor("#141b2d")

    ages   = [r["age"] for r in results]
    worsts = [r["worst"] for r in results]
    bests  = [r["best"] for r in results]
    sames  = [r["same"] for r in results]

    ax.plot(ages, worsts, color="#ef4444", lw=3, label="Worsening Habits (BP+ / Smoking+)")
    ax.plot(ages, sames,  color="#fbbf24", lw=2, linestyle="--", label="Stay as You Are (Only Age)")
    ax.plot(ages, bests,  color="#10b981", lw=3, label="Optimal Habits (Weight- / Quit Smoking)")
    
    ax.axhline(50, color="white", alpha=0.1, linestyle=":")
    ax.set_ylim(-2, 102); ax.tick_params(colors="#94a3b8")
    ax.set_title("10-Year Clinical Action Simulation", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Patient Age", color="#cbd5e1"); ax.set_ylabel("CHD Risk Probability (%)", color="#cbd5e1")
    ax.legend(facecolor="#1e2a3a", edgecolor="#2d3f60", labelcolor="white", loc="upper left")
    ax.grid(alpha=0.1, color="white")
    fig.tight_layout()

    return {"sim_img": fig_to_b64(fig), "summary": summary}
