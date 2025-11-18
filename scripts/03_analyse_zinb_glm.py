import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import patsy
import warnings

# --- WARNUNGEN UNTERDRÜCKEN ---
# Wir ignorieren HessianInversionWarning, da Nelder-Mead (Schritt 1)
# keine Standardfehler berechnen kann (das ist erwartet).
from statsmodels.tools.sm_exceptions import HessianInversionWarning, ConvergenceWarning

warnings.simplefilter('ignore', category=HessianInversionWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

print("--- Starte ZINB & GLM Analyse (Final & Clean) ---")

# Pfade setzen
script_dir = os.path.dirname(os.path.abspath(__file__))
# Ggf. anpassen, falls dein 'results' Ordner woanders liegt
base_path = os.path.join(script_dir, '..', 'results')

# ---------------------------------------------------------
# 1. DATEN LADEN & VORBEREITEN
# ---------------------------------------------------------
print("Lade Daten...")
try:
    df_de = pd.read_csv(os.path.join(base_path, 'germany_analysis_ready.csv'), low_memory=False)
    df_fr = pd.read_csv(os.path.join(base_path, 'france_analysis_ready.csv'), low_memory=False)
    df_ee = pd.read_csv(os.path.join(base_path, 'estonia_analysis_ready.csv'), low_memory=False)
except FileNotFoundError:
    print("KRITISCHER FEHLER: Dateien nicht gefunden. Bitte Pfade prüfen.")
    exit()

# Länder-Label setzen
df_de['country'] = 'Germany'
df_fr['country'] = 'France'
df_ee['country'] = 'Estonia'

# Zusammenfügen
df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)

# Arbeitskopie
reg_df = df_final.copy()

# --- Bereinigung & Feature Engineering ---

# 1. Extremwerte filtern
reg_df = reg_df[reg_df['total_bids'] < 100]

# 2. Duration
reg_df['duration_days'] = pd.to_numeric(reg_df['duration_days'], errors='coerce')
reg_df = reg_df[reg_df['duration_days'] > 0]
p99_dur = reg_df['duration_days'].quantile(0.99)
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99_dur)
reg_df['z_duration'] = (
                               reg_df['duration_days_capped'] - reg_df['duration_days_capped'].mean()
                       ) / reg_df['duration_days_capped'].std()

# 3. Tender Value
reg_df['tender_value'] = pd.to_numeric(reg_df['tender_value'], errors='coerce')
reg_df = reg_df[reg_df['tender_value'] > 0]
reg_df['log_tender_value'] = np.log(reg_df['tender_value'])
reg_df['z_value'] = (
                            reg_df['log_tender_value'] - reg_df['log_tender_value'].mean()
                    ) / reg_df['log_tender_value'].std()

# 4. H2 Variable
reg_df['sme_share'] = np.where(
    reg_df['total_bids'] > 0,
    reg_df['sme_bids'] / reg_df['total_bids'],
    np.nan
)
reg_df['sme_share'] = reg_df['sme_share'].clip(0, 1)

# 5. Konstante für Inflation
reg_df['const'] = 1

print(f"Basis-Datensatzgröße: {len(reg_df)}")

# ---------------------------------------------------------
# 2. MODELL 1: ZINB (H1 & H3a - Wettbewerb)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MODELL 1: ZINB (total_bids) - Zwei-Schritt-Optimierung")
print("=" * 60)

zinb_vars = ['total_bids', 'z_duration', 'country', 'z_value',
             'procurement_method', 'procurement_category', 'year', 'const']
reg_df_z1 = reg_df.dropna(subset=zinb_vars).copy()
print(f"Stichprobe ZINB: {len(reg_df_z1)}")

# Formel
count_formula = (
    "total_bids ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(procurement_category) + C(year)"
)

y_count, X_count = patsy.dmatrices(count_formula, data=reg_df_z1, return_type='dataframe')
X_infl = reg_df_z1[['const']]

zinb_model_instance = ZeroInflatedNegativeBinomialP(
    endog=y_count,
    exog=X_count,
    exog_infl=X_infl,
    inflation='logit'
)

try:
    # Schritt 1: Nelder-Mead (findet Koeffizienten, aber keine p-Werte -> Warnung ignoriert)
    print("Berechne Startwerte (Nelder-Mead)...")
    res_nm = zinb_model_instance.fit(maxiter=10000, method='nm', disp=0)

    # Schritt 2: BFGS (nutzt Startwerte, berechnet p-Werte)
    print("Berechne finales Modell (BFGS)...")
    zinb_result = zinb_model_instance.fit(maxiter=10000, method='bfgs', start_params=res_nm.params, disp=0)

    print(zinb_result.summary())

except Exception as e:
    print(f"Fehler: {e}")
    # Fallback falls alles scheitert
    try:
        print("Versuche Newton-CG...")
        zinb_result = zinb_model_instance.fit(maxiter=10000, method='newton', start_params=res_nm.params, disp=0)
        print(zinb_result.summary())
    except:
        print("Konnte Modell nicht berechnen.")

# ---------------------------------------------------------
# 3. MODELL 2: GLM (H2 & H3b - KMU)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MODELL 2: Fractional Logit (sme_share)")
print("=" * 60)

reg_df_h2 = reg_df.dropna(subset=['sme_share'] + zinb_vars).copy()
epsilon = 1e-6
reg_df_h2['sme_share_safe'] = reg_df_h2['sme_share'].clip(epsilon, 1 - epsilon)

print(f"Stichprobe GLM: {len(reg_df_h2)}")

glm_formula = (
    "sme_share_safe ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(procurement_category) + C(year)"
)

try:
    glm_model = smf.glm(
        formula=glm_formula,
        data=reg_df_h2,
        family=sm.families.Binomial(link=sm.families.links.logit())
    ).fit(cov_type='HC0')

    print(glm_model.summary())

except Exception as e:
    print(f"Fehler beim GLM-Fitting: {e}")

print("\n--- Analyse beendet ---")