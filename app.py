import os
import io
import zipfile
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.io.pwscf import PWInput
import shap

st.set_page_config(
    page_title="MaterialsGPT Africa | Team P03 AZIB Discovery",
    page_icon="🔋",
    layout="wide"
)

DRIVE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DRIVE_DIR, "materialsgpt_p03_azib.db")
FEATURES_PATH = os.path.join(DRIVE_DIR, "WP2_engineered_features.csv")

PRETTY_NAMES = {
    "dopant_atomic_pct": "Dopant Conc. (at.%)",
    "intercalant_stoich": "Intercalant Stoich. (x)",
    "molar_mass_g_mol": "Molar Mass (g/mol)",
    "theoretical_cap_2e_mah_g": "Theoretical 2e- Cap. (mAh/g)",
    "mean_electronegativity": "Mean Electronegativity",
    "electronegativity_spread": "Electronegativity Spread",
    "dopant_radius_strain_pct": "Dopant Radius Strain (%)",
    "effective_spacing_A": "Tunnel / Layer Spacing (A)",
    "base_mn_o_bond_A": "Baseline Mn-O Bond (A)",
    "base_distortion_idx": "Octahedral Distortion Index",
    "base_band_gap_eV": "Baseline Band Gap (eV)",
    "mn_additive_conc_m": "Mn2+ Additive Conc. (M)",
    "current_density_a_g": "Current Density (A/g)"
}

FEATURE_COLS = list(PRETTY_NAMES.keys())

BASE_CIFS = {
    "alpha": os.path.join(DRIVE_DIR, "MNO2-ALPHA-PRISTINE-NONE-001_mp-19395.cif"),
    "beta":  os.path.join(DRIVE_DIR, "MNO2-BETA-PRISTINE-NONE-001_mp-510408.cif"),
    "gamma": os.path.join(DRIVE_DIR, "MNO2-GAMMA-PRISTINE-NONE-001_mp-19326.cif"),
    "delta": os.path.join(DRIVE_DIR, "MNO2-DELTA-PRISTINE-NONE-001_mp-25424.cif"),
}

PSEUDOS = {
    "Mn": "Mn.pbe-spn-kjpaw_psl.0.3.1.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ni": "Ni.pbe-spn-kjpaw_psl.0.1.4.UPF",
    "K":  "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Na": "Na.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "O":  "O.pbe-n-kjpaw_psl.0.1.UPF"
}

def build_workstation_dft_zip(row):
    poly = row.get("polymorph")
    cif_path = BASE_CIFS.get(poly)
    if not cif_path or not os.path.exists(cif_path):
        return None, 0, "CIF missing"

    struct = Structure.from_file(cif_path).copy()
    sc_dims = (1, 1, 2) if poly in ("alpha", "beta", "gamma") else (3, 3, 1)
    struct.make_supercell(sc_dims)

    dopant = row.get("dopant_element")
    if pd.notna(dopant) and str(dopant) not in ("None", ""):
        mn_sites = tuple(i for i, s in enumerate(struct) if s.specie.symbol == "Mn")
        if len(mn_sites) > 0:
            struct.replace(mn_sites.__getitem__(0), str(dopant))

    ion = row.get("intercalant_ion")
    if pd.notna(ion) and str(ion) not in ("None", ""):
        ion_symbol = str(ion).replace("+", "")
        if poly == "delta":
            struct.append(ion_symbol, (0.167, 0.167, 0.50), coords_are_cartesian=False)
            struct.append(ion_symbol, (0.667, 0.667, 0.50), coords_are_cartesian=False)
        else:
            struct.append(ion_symbol, (0.0, 0.0, 0.25), coords_are_cartesian=False)
            struct.append(ion_symbol, (0.5, 0.5, 0.25), coords_are_cartesian=False)

    buf = io.BytesIO()
    prefix = row.get("material_id")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}_supercell.cif", struct.to(fmt="cif"))
        zf.writestr(f"{prefix}.POSCAR", str(Poscar(struct)))

        present_els = {el.symbol: PSEUDOS.get(el.symbol) for el in struct.composition.elements if el.symbol in PSEUDOS}
        qe_in = PWInput(
            structure=struct,
            pseudo=present_els,
            control={"calculation": "vc-relax", "prefix": prefix, "outdir": "./tmp", "pseudo_dir": "./pseudo", "etot_conv_thr": 1e-5, "forc_conv_thr": 1e-4},
            system={"ecutwfc": 60, "ecutrho": 480, "occupations": "smearing", "smearing": "gaussian", "degauss": 0.01, "nspin": 2},
            electrons={"conv_thr": 1e-7, "mixing_beta": 0.3},
            ions={"ion_dynamics": "bfgs"},
            cell={"cell_dynamics": "bfgs"},
            kpoints_grid=(3, 3, 4) if poly != "delta" else (3, 3, 2)
        )
        zf.writestr(f"{prefix}_qe_relax.in", str(qe_in))
    buf.seek(0)
    return buf, len(struct), struct.composition.reduced_formula

@st.cache_data
def load_platform_data():
    conn = sqlite3.connect(DB_PATH)
    try:
        materials = pd.read_sql_query("SELECT * FROM materials", conn)
        structures = pd.read_sql_query("SELECT * FROM structures", conn)
        descriptors = pd.read_sql_query("SELECT * FROM descriptors", conn)
        experiments = pd.read_sql_query("SELECT * FROM experiments", conn)
        computations = pd.read_sql_query("SELECT * FROM computations", conn)
    finally:
        conn.close()

    features_df = pd.read_csv(FEATURES_PATH)
    for col in ("dopant_atomic_pct", "intercalant_stoich"):
        features_df[col] = features_df[col].fillna(0.0)

    train_df = pd.merge(experiments, features_df, on="material_id", how="inner")
    X_train = train_df.loc[:, FEATURE_COLS]
    rf_cap = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, train_df.get("specific_capacity_mah_g"))
    rf_ret = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, train_df.get("capacity_retention_pct"))

    proposed = features_df.loc[features_df.get("evidence_status") == "proposed"].reset_index(drop=True)
    proposed["mn_additive_conc_m"] = 0.1
    proposed["current_density_a_g"] = 0.2
    X_pred = proposed.loc[:, FEATURE_COLS]

    tree_cap = np.array(tuple(t.predict(X_pred.values) for t in rf_cap.estimators_))
    tree_ret = np.array(tuple(t.predict(X_pred.values) for t in rf_ret.estimators_))

    proposed["pred_capacity_mah_g"] = tree_cap.mean(axis=0).round(2)
    proposed["pred_capacity_std"] = tree_cap.std(axis=0).round(2)
    proposed["pred_retention_pct"] = tree_ret.mean(axis=0).round(2)

    explainer = shap.TreeExplainer(rf_cap)
    shap_vals = explainer.shap_values(X_pred)

    return materials, structures, descriptors, experiments, computations, features_df, proposed, shap_vals

materials, structures, descriptors, experiments, computations, features_df, proposed_df, shap_vals = load_platform_data()

st.title("🔋 MaterialsGPT Africa — Team P03 Discovery Platform")
st.caption("AI-Guided Discovery of Affordable MnO₂-Based Cathode Materials for Aqueous Zinc-Ion Batteries (AZIBs)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cataloged Materials", len(materials))
k2.metric("Verified Experimental Records", len(experiments))
k3.metric("Baseline DFT Polymorphs", len(computations))
k4.metric("Proposed Virtual Candidates", len(proposed_df))

st.divider()

st.sidebar.header("🔍 Candidate Exploration Filters")
selected_polymorphs = st.sidebar.multiselect("MnO₂ Polymorph", ("alpha", "beta", "gamma", "delta"), default=("alpha", "beta", "gamma", "delta"))
selected_dopants = st.sidebar.multiselect("Substitutional Dopant", ("None", "Fe", "Ti", "Ni"), default=("None", "Fe", "Ti", "Ni"))
selected_ions = st.sidebar.multiselect("Pre-Intercalation Ion", ("None", "K+", "Na+"), default=("None", "K+", "Na+"))
selected_evidence = st.sidebar.multiselect("Evidence Status", ("both", "experimental", "proposed"), default=("both", "experimental", "proposed"))

st.sidebar.divider()
st.sidebar.header("⚖️ Multi-Objective Ranking Weights")
w_cap = st.sidebar.slider("Weight: Specific Capacity", 0.0, 1.0, 0.50, 0.05)
w_ret = round(1.0 - w_cap, 2)
st.sidebar.write(f"**Weight: Cycling Stability:** `{w_ret:.2f}`")

filtered = features_df.copy()
filtered["dopant_clean"] = filtered.get("dopant_element").fillna("None")
filtered["ion_clean"] = filtered.get("intercalant_ion").fillna("None")
filtered = filtered.loc[
    (filtered.get("polymorph").isin(selected_polymorphs)) &
    (filtered.get("dopant_clean").isin(selected_dopants)) &
    (filtered.get("ion_clean").isin(selected_ions)) &
    (filtered.get("evidence_status").isin(selected_evidence))
]

tab1, tab2, tab3, tab4 = st.tabs((
    "📊 Candidate Explorer",
    "🏆 AI Ranking, SHAP & DFT Export",
    "🧪 Experimental vs. DFT Evidence",
    "🧭 Evidence Gaps"
))

with tab1:
    st.subheader("Filtered MnO₂ Candidate Matrix")
    display_cols = list((
        "material_id", "formula", "polymorph", "modification_type",
        "dopant_clean", "dopant_atomic_pct", "ion_clean", "intercalant_stoich",
        "evidence_status", "theoretical_cap_2e_mah_g", "dopant_radius_strain_pct", "effective_spacing_A"
    ))
    st.dataframe(filtered.loc[:, display_cols], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Ranked Virtual Co-Modified Candidates (ML Predictions)")
    st.info("⚠️ Transparency Notice: Values below are Random Forest predictions at 0.2 A/g (2M ZnSO₄ + 0.1M MnSO₄) and are not yet experimentally synthesized.")

    ranked = proposed_df.copy()
    ranked["dopant_clean"] = ranked.get("dopant_element").fillna("None")
    ranked["ion_clean"] = ranked.get("intercalant_ion").fillna("None")
    ranked = ranked.loc[
        (ranked.get("polymorph").isin(selected_polymorphs)) &
        (ranked.get("dopant_clean").isin(selected_dopants)) &
        (ranked.get("ion_clean").isin(selected_ions))
    ].copy()

    if len(ranked) == 0:
        st.warning("No proposed candidates match your current sidebar filters.")
    else:
        ranked["dynamic_score"] = (w_cap * (ranked.get("pred_capacity_mah_g") / 308.0) + w_ret * (ranked.get("pred_retention_pct") / 100.0)).round(4)
        ranked = ranked.sort_values("dynamic_score", ascending=False)
        rank_cols = list(("material_id", "formula", "polymorph", "dopant_element", "intercalant_ion", "pred_capacity_mah_g", "pred_capacity_std", "pred_retention_pct", "dynamic_score"))
        st.dataframe(ranked.loc[:, rank_cols], use_container_width=True, hide_index=True)

