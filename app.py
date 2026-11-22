import os
import io
import zipfile
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from pymatgen.core import Structure, Composition
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

IONIC_RADII = {"Mn": 0.53, "Fe": 0.645, "Ti": 0.605, "Ni": 0.69, "K+": 1.38, "Na+": 1.02}
FARADAY_2E = 53603.0

def compute_features_from_db(materials, structures, descriptors, computations):
    base_m = materials.loc[materials["modification_type"] == "pristine", ["material_id", "polymorph"]]
    base_merged = pd.merge(base_m, structures[["material_id", "structure_id"]], on="material_id")
    base_merged = pd.merge(base_merged, descriptors, on="structure_id")
    base_merged = pd.merge(base_merged, computations, on="structure_id")
    base_map = base_merged.set_index("polymorph").to_dict(orient="index")

    rows = []
    for _, row in materials.iterrows():
        comp = Composition(str(row["formula"]))
        molar_mass = float(comp.weight)
        theo_cap = FARADAY_2E / molar_mass
        mean_en = sum(el.X * amt for el, amt in comp.items()) / comp.num_atoms
        en_spread = max(el.X for el in comp.elements) - min(el.X for el in comp.elements)

        dop = row["dopant_element"]
        dop_pct = float(row["dopant_atomic_pct"]) if pd.notna(row["dopant_atomic_pct"]) else 0.0
        strain = (((IONIC_RADII[dop] - IONIC_RADII["Mn"]) / IONIC_RADII["Mn"]) * 100.0 * (dop_pct / 100.0)) if (pd.notna(dop) and dop in IONIC_RADII) else 0.0

        b = base_map.get(row["polymorph"], {})
        b_space = float(b.get("tunnel_or_interlayer_spacing", 4.60))
        ion = row["intercalant_ion"]
        i_stoich = float(row["intercalant_stoich"]) if pd.notna(row["intercalant_stoich"]) else 0.0
        boost = ((IONIC_RADII[ion] * 0.35) * (i_stoich / 0.15) if row["polymorph"] == "delta" else 0.05) if (pd.notna(ion) and ion in IONIC_RADII) else 0.0

        rows.append({
            "molar_mass_g_mol": round(molar_mass, 3),
            "theoretical_cap_2e_mah_g": round(theo_cap, 2),
            "mean_electronegativity": round(mean_en, 4),
            "electronegativity_spread": round(en_spread, 4),
            "dopant_radius_strain_pct": round(strain, 4),
            "effective_spacing_A": round(b_space + boost, 3),
            "base_mn_o_bond_A": float(b.get("avg_mn_o_bond_length", 1.89)),
            "base_distortion_idx": float(b.get("polyhedral_distortion_index", 0.0045)),
            "base_band_gap_eV": float(b.get("band_gap_ev", 1.20))
        })
    return pd.concat([materials.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

def build_workstation_dft_zip(row):
    poly = row["polymorph"]
    cif_path = BASE_CIFS.get(poly)
    if not cif_path or not os.path.exists(cif_path):
        return None, 0, "CIF missing"

    struct = Structure.from_file(cif_path).copy()
    sc_dims = (1, 1, 2) if poly in ("alpha", "beta", "gamma") else (3, 3, 1)
    struct.make_supercell(sc_dims)

    dopant = row["dopant_element"]
    if pd.notna(dopant) and str(dopant) not in ("None", ""):
        mn_sites = [i for i, s in enumerate(struct) if s.specie.symbol == "Mn"]
        if mn_sites:
            struct.replace(mn_sites[0], str(dopant))

    ion = row["intercalant_ion"]
    if pd.notna(ion) and str(ion) not in ("None", ""):
        ion_symbol = str(ion).replace("+", "")
        if poly == "delta":
            struct.append(ion_symbol, (0.167, 0.167, 0.50), coords_are_cartesian=False)
            struct.append(ion_symbol, (0.667, 0.667, 0.50), coords_are_cartesian=False)
        else:
            struct.append(ion_symbol, (0.0, 0.0, 0.25), coords_are_cartesian=False)
            struct.append(ion_symbol, (0.5, 0.5, 0.25), coords_are_cartesian=False)

    buf = io.BytesIO()
    prefix = row["material_id"]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}_supercell.cif", struct.to(fmt="cif"))
        zf.writestr(f"{prefix}.POSCAR", str(Poscar(struct)))

        present_els = {el.symbol: PSEUDOS[el.symbol] for el in struct.composition.elements if el.symbol in PSEUDOS}
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

    if os.path.exists(FEATURES_PATH):
        features_df = pd.read_csv(FEATURES_PATH)
    else:
        features_df = compute_features_from_db(materials, structures, descriptors, computations)

    for col in ("dopant_atomic_pct", "intercalant_stoich"):
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0.0)

    train_df = pd.merge(experiments, features_df, on="material_id", how="inner")
    X_train = train_df[FEATURE_COLS]
    rf_cap = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, train_df["specific_capacity_mah_g"])
    rf_ret = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, train_df["capacity_retention_pct"])

    proposed = features_df[features_df["evidence_status"] == "proposed"].reset_index(drop=True)
    proposed["mn_additive_conc_m"] = 0.1
    proposed["current_density_a_g"] = 0.2
    X_pred = proposed[FEATURE_COLS]

    tree_cap = np.array([t.predict(X_pred.values) for t in rf_cap.estimators_])
    tree_ret = np.array([t.predict(X_pred.values) for t in rf_ret.estimators_])

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
selected_polymorphs = st.sidebar.multiselect("MnO₂ Polymorph", ["alpha", "beta", "gamma", "delta"], default=["alpha", "beta", "gamma", "delta"])
selected_dopants = st.sidebar.multiselect("Substitutional Dopant", ["None", "Fe", "Ti", "Ni"], default=["None", "Fe", "Ti", "Ni"])
selected_ions = st.sidebar.multiselect("Pre-Intercalation Ion", ["None", "K+", "Na+"], default=["None", "K+", "Na+"])
selected_evidence = st.sidebar.multiselect("Evidence Status", ["both", "experimental", "proposed"], default=["both", "experimental", "proposed"])

st.sidebar.divider()
st.sidebar.header("⚖️ Multi-Objective Ranking Weights")
w_cap = st.sidebar.slider("Weight: Specific Capacity", 0.0, 1.0, 0.50, 0.05)
