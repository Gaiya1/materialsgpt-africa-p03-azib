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

def get_fallback_records():
    mats = pd.DataFrame([
        {"material_id": "MNO2-ALPHA-DOPED-FE-001",    "formula": "Fe0.05Mn0.95O2", "polymorph": "alpha", "modification_type": "doped",            "dopant_element": "Fe",  "dopant_atomic_pct": 5.0,  "intercalant_ion": None,  "intercalant_stoich": None, "evidence_status": "experimental"},
        {"material_id": "MNO2-ALPHA-DOPED-TI-001",    "formula": "Ti0.04Mn0.96O2", "polymorph": "alpha", "modification_type": "doped",            "dopant_element": "Ti",  "dopant_atomic_pct": 4.0,  "intercalant_ion": None,  "intercalant_stoich": None, "evidence_status": "experimental"},
        {"material_id": "MNO2-DELTA-DOPED-NI-001",    "formula": "Ni0.06Mn0.94O2", "polymorph": "delta", "modification_type": "doped",            "dopant_element": "Ni",  "dopant_atomic_pct": 6.0,  "intercalant_ion": None,  "intercalant_stoich": None, "evidence_status": "experimental"},
        {"material_id": "MNO2-ALPHA-INTERCAL-K-001",  "formula": "K0.14MnO2",      "polymorph": "alpha", "modification_type": "pre-intercalated", "dopant_element": None,  "dopant_atomic_pct": None, "intercalant_ion": "K+",  "intercalant_stoich": 0.14, "evidence_status": "experimental"},
        {"material_id": "MNO2-DELTA-INTERCAL-NA-001", "formula": "Na0.20MnO2",     "polymorph": "delta", "modification_type": "pre-intercalated", "dopant_element": None,  "dopant_atomic_pct": None, "intercalant_ion": "Na+", "intercalant_stoich": 0.20, "evidence_status": "experimental"},
    ])
    exps = pd.DataFrame([
        {"experiment_id": "EXP-001", "material_id": "MNO2-ALPHA-DOPED-FE-001",    "source_id": "SRC-LIT-007", "electrolyte_salt": "2M ZnSO4", "mn_additive_conc_m": 0.1, "voltage_window_min": 0.8, "voltage_window_max": 1.8, "mass_loading_mg_cm2": 1.5, "current_density_a_g": 0.2, "initial_discharge_cap_mah_g": 318.0, "specific_capacity_mah_g": 305.0, "cycle_number_tested": 500,  "capacity_retention_pct": 92.4, "coulombic_efficiency_pct": 99.6},
        {"experiment_id": "EXP-002", "material_id": "MNO2-ALPHA-DOPED-TI-001",    "source_id": "SRC-LIT-008", "electrolyte_salt": "2M ZnSO4", "mn_additive_conc_m": 0.1, "voltage_window_min": 0.8, "voltage_window_max": 1.8, "mass_loading_mg_cm2": 1.6, "current_density_a_g": 0.2, "initial_discharge_cap_mah_g": 295.0, "specific_capacity_mah_g": 284.0, "cycle_number_tested": 500,  "capacity_retention_pct": 90.1, "coulombic_efficiency_pct": 99.4},
        {"experiment_id": "EXP-003", "material_id": "MNO2-DELTA-DOPED-NI-001",    "source_id": "SRC-LIT-009", "electrolyte_salt": "2M ZnSO4", "mn_additive_conc_m": 0.1, "voltage_window_min": 0.8, "voltage_window_max": 1.8, "mass_loading_mg_cm2": 1.4, "current_density_a_g": 0.2, "initial_discharge_cap_mah_g": 310.0, "specific_capacity_mah_g": 296.0, "cycle_number_tested": 400,  "capacity_retention_pct": 89.5, "coulombic_efficiency_pct": 99.3},
        {"experiment_id": "EXP-004", "material_id": "MNO2-ALPHA-INTERCAL-K-001",  "source_id": "SRC-LIT-005", "electrolyte_salt": "2M ZnSO4", "mn_additive_conc_m": 0.1, "voltage_window_min": 0.8, "voltage_window_max": 1.8, "mass_loading_mg_cm2": 1.5, "current_density_a_g": 0.2, "initial_discharge_cap_mah_g": 312.0, "specific_capacity_mah_g": 299.0, "cycle_number_tested": 1000, "capacity_retention_pct": 94.0, "coulombic_efficiency_pct": 99.7},
        {"experiment_id": "EXP-005", "material_id": "MNO2-DELTA-INTERCAL-NA-001", "source_id": "SRC-LIT-006", "electrolyte_salt": "2M ZnSO4", "mn_additive_conc_m": 0.1, "voltage_window_min": 0.8, "voltage_window_max": 1.8, "mass_loading_mg_cm2": 1.5, "current_density_a_g": 0.2, "initial_discharge_cap_mah_g": 304.0, "specific_capacity_mah_g": 290.0, "cycle_number_tested": 800,  "capacity_retention_pct": 91.8, "coulombic_efficiency_pct": 99.5},
    ])
    return mats, exps

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

def compute_local_attributions(rf_model, X_train_df, X_pred_df):
    importances = rf_model.feature_importances_
    means = X_train_df.mean().to_numpy()
    stds = X_train_df.std().replace(0, 1.0).to_numpy()
    z_scores = (X_pred_df.to_numpy() - means) / stds
    return z_scores * importances * 10.0

def build_workstation_dft_zip(row):
    poly = str(row["polymorph"])
    cif_path = BASE_CIFS.get(poly)
    if not cif_path or not os.path.exists(cif_path):
        return None, 0, "CIF missing"

    struct = Structure.from_file(cif_path).copy()
    sc_dims = (1, 1, 2) if poly in ("alpha", "beta", "gamma") else (3, 3, 1)
    struct.make_supercell(sc_dims)

    dopant = row["dopant_element"]
    if pd.notna(dopant) and str(dopant) not in ("None", "", "nan"):
        mn_sites = tuple(i for i, s in enumerate(struct) if s.specie.symbol == "Mn")
        if len(mn_sites) > 0:
            struct.replace(mn_sites.__getitem__(0), str(dopant))

    ion = row["intercalant_ion"]
    if pd.notna(ion) and str(ion) not in ("None", "", "nan"):
        ion_symbol = str(ion).replace("+", "")
        if poly == "delta":
            struct.append(ion_symbol, (0.167, 0.167, 0.50), coords_are_cartesian=False)
            struct.append(ion_symbol, (0.667, 0.667, 0.50), coords_are_cartesian=False)
        else:
            struct.append(ion_symbol, (0.0, 0.0, 0.25), coords_are_cartesian=False)
            struct.append(ion_symbol, (0.5, 0.5, 0.25), coords_are_cartesian=False)

    buf = io.BytesIO()
    prefix = str(row["material_id"])
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}_supercell.cif", struct.to(fmt="cif"))
        zf.writestr(f"{prefix}.POSCAR", str(Poscar(struct)))

