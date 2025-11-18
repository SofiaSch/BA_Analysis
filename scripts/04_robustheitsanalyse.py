import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import patsy
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning, ConvergenceWarning

# Warnungen unterdrücken
warnings.simplefilter('ignore', category=HessianInversionWarning)
warnings.simplefilter('ignore', category=ConvergenceWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

print("--- Starte Robustheits-Checks (ZINB & GLM) ---")

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', 'results')

# 1. DATEN LADEN & VORBEREITEN (Identisch zur Hauptanalyse)
print("Lade Daten...")
try:
    df_de = pd.read_csv(os.path.join(base_path, 'germany_analysis_ready.csv'), low_memory=False)
    df_fr = pd.read_csv(os.path.join(base_path, 'france_analysis_ready.csv'), low_memory=False)
    df_ee = pd.read_csv(os.path.join(base_path, 'estonia_analysis_ready.csv'), low_memory=False)
except FileNotFoundError:
    print("Fehler: Daten nicht gefunden.")
    exit()

df_de['country'] = 'Germany'
df_fr['country'] = 'France'
df_ee['country'] = 'Estonia'
df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
reg_df = df_final.copy()

# Standard-Bereinigung
reg_df = reg_df[reg_df['total_bids'] < 100]
reg_df['duration_days'] = pd.to_numeric(reg_df['duration_days'], errors='coerce')
reg_df = reg_df[reg_df['duration_days'] > 0]
p99 = reg_df['duration_days'].quantile(0.99)
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99)
reg_df['z_duration'] = (reg_df['duration_days_capped'] - reg_df['duration_days_capped'].mean()) / reg_df[
    'duration_days_capped'].std()

reg_df['tender_value'] = pd.to_numeric(reg_df['tender_value'], errors='coerce')
reg_df = reg_df[reg_df['tender_value'] > 0]
reg_df['log_tender_value'] = np.log(reg_df['tender_value'])
reg_df['z_value'] = (reg_df['log_tender_value'] - reg_df['log_tender_value'].mean()) / reg_df['log_tender_value'].std()
reg_df['const'] = 1

# -------------------------------------------------------------------
# ROBUSTHEITS-CHECK 1: Zusätzliche Kontrolle 'award_criteria'
# Wir prüfen, ob das Ergebnis stabil bleibt, wenn wir das Vergabekriterium (Preis vs. Qualität) aufnehmen.
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("ROBUSTHEIT 1: ZINB mit 'award_criteria'")
print("=" * 60)

# Wir nehmen nur Daten, wo award_criteria vorhanden ist
vars_r1 = ['total_bids', 'z_duration', 'country', 'z_value', 'procurement_method',
           'procurement_category', 'year', 'const', 'award_criteria']
df_r1 = reg_df.dropna(subset=vars_r1).copy()

# Formel erweitert um C(award_criteria)
formula_r1 = (
    "total_bids ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(procurement_category) + C(year) + C(award_criteria)"
)

try:
    y_r1, X_r1 = patsy.dmatrices(formula_r1, data=df_r1, return_type='dataframe')
    X_infl_r1 = df_r1[['const']]

    # 1. Schritt: Startwerte finden
    model_temp = ZeroInflatedNegativeBinomialP(endog=y_r1, exog=X_r1, exog_infl=X_infl_r1, inflation='logit')
    res_temp = model_temp.fit(maxiter=5000, method='nm', disp=0)

    # 2. Schritt: Finales Fitting
    res_r1 = model_temp.fit(maxiter=5000, method='bfgs', start_params=res_temp.params, disp=0)

    # Zeige nur die relevanten Variablen (Dauer & Interaktionen)
    print(res_r1.summary().tables[1])
    print("\n-> Wenn z_duration und Interaktionen ähnlich wie im Hauptmodell sind: ROBUST!")

except Exception as e:
    print(f"Fehler bei Robustheit 1: {e}")

# -------------------------------------------------------------------
# ROBUSTHEITS-CHECK 2: Nur 'Services' (Dienstleistungen)
# Wir prüfen, ob die Effekte in der größten Kategorie auch alleine gelten.
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("ROBUSTHEIT 2: ZINB nur für 'Services'")
print("=" * 60)

df_r2 = reg_df[reg_df['procurement_category'] == 'services'].copy()
# Bereinigen (ohne procurement_category in der Formel, da konstant)
vars_r2 = ['total_bids', 'z_duration', 'country', 'z_value', 'procurement_method', 'year', 'const']
df_r2 = df_r2.dropna(subset=vars_r2)

formula_r2 = (
    "total_bids ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(year)"
)

try:
    y_r2, X_r2 = patsy.dmatrices(formula_r2, data=df_r2, return_type='dataframe')
    X_infl_r2 = df_r2[['const']]

    model_temp2 = ZeroInflatedNegativeBinomialP(endog=y_r2, exog=X_r2, exog_infl=X_infl_r2, inflation='logit')
    res_temp2 = model_temp2.fit(maxiter=5000, method='nm', disp=0)
    res_r2 = model_temp2.fit(maxiter=5000, method='bfgs', start_params=res_temp2.params, disp=0)

    print(res_r2.summary().tables[1])

except Exception as e:
    print(f"Fehler bei Robustheit 2: {e}")

print("\n--- Robustheits-Checks beendet ---")