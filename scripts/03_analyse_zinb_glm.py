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

print("--- Starte ZINB & GLM Analyse (inkl. Stichproben-Statistik) ---")

# Pfade setzen
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', 'results')

# ---------------------------------------------------------
# 1. DATEN LADEN & URSPRUNGS-ZÄHLUNG
# ---------------------------------------------------------
print("Lade Daten...")
try:
    df_de = pd.read_csv(os.path.join(base_path, 'germany_analysis_ready.csv'), low_memory=False)
    df_fr = pd.read_csv(os.path.join(base_path, 'france_analysis_ready.csv'), low_memory=False)
    df_ee = pd.read_csv(os.path.join(base_path, 'estonia_analysis_ready.csv'), low_memory=False)
except FileNotFoundError:
    print("KRITISCHER FEHLER: Dateien nicht gefunden. Bitte Pfade prüfen.")
    exit()

# Urspungsgrößen speichern
n_de_raw = len(df_de)
n_fr_raw = len(df_fr)
n_ee_raw = len(df_ee)

# Länder-Label setzen
df_de['country'] = 'Germany'
df_fr['country'] = 'France'
df_ee['country'] = 'Estonia'

# Zusammenfügen
df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)

# Arbeitskopie
reg_df = df_final.copy()

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING & BEREINIGUNG
# ---------------------------------------------------------

# A. Technische Bereinigung (Extremwerte Bids)
reg_df = reg_df[reg_df['total_bids'] < 100]

# B. Duration
reg_df['duration_days'] = pd.to_numeric(reg_df['duration_days'], errors='coerce')
reg_df = reg_df[reg_df['duration_days'] > 0]
p99_dur = reg_df['duration_days'].quantile(0.99)
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99_dur)
reg_df['z_duration'] = (
    reg_df['duration_days_capped'] - reg_df['duration_days_capped'].mean()
) / reg_df['duration_days_capped'].std()

# C. Tender Value (Hier entstehen die meisten NaNs!)
reg_df['tender_value'] = pd.to_numeric(reg_df['tender_value'], errors='coerce')
reg_df = reg_df[reg_df['tender_value'] > 0]
reg_df['log_tender_value'] = np.log(reg_df['tender_value'])
reg_df['z_value'] = (
    reg_df['log_tender_value'] - reg_df['log_tender_value'].mean()
) / reg_df['log_tender_value'].std()

# D. H2 Variable
reg_df['sme_share'] = np.where(
    reg_df['total_bids'] > 0,
    reg_df['sme_bids'] / reg_df['total_bids'],
    np.nan
)
reg_df['sme_share'] = reg_df['sme_share'].clip(0, 1)

# E. Konstante
reg_df['const'] = 1

# ---------------------------------------------------------
# 3. FINALER STICHPROBEN-FILTER (LISTWISE DELETION)
# ---------------------------------------------------------
# Das sind die Variablen, die wir für Modell 1 (H1/H3) zwingend brauchen
model_vars = ['total_bids', 'z_duration', 'country', 'z_value',
              'procurement_method', 'procurement_category', 'year', 'const']

# Wir erstellen den finalen Datensatz für die Analyse
df_model = reg_df.dropna(subset=model_vars).copy()

# Zählung pro Land im finalen Datensatz
counts_final = df_model['country'].value_counts()

# ---------------------------------------------------------
# 4. AUSGABE DER STATISTIK FÜR DEINE ARBEIT
# ---------------------------------------------------------
print("\n" + "="*60)
print("STATISTIK FÜR KAPITEL 4 (METHODIK)")
print("="*60)
print(f"{'Land':<15} | {'Rohdaten':<10} | {'Final (Modell)':<15} | {'Verlust (%)':<10}")
print("-" * 60)

for country, n_raw in [('Germany', n_de_raw), ('France', n_fr_raw), ('Estonia', n_ee_raw)]:
    n_final = counts_final.get(country, 0)
    loss_pct = ((n_raw - n_final) / n_raw) * 100
    print(f"{country:<15} | {n_raw:<10} | {n_final:<15} | {loss_pct:.1f}%")

print("-" * 60)
print(f"{'GESAMT':<15} | {n_de_raw+n_fr_raw+n_ee_raw:<10} | {len(df_model):<15} | {((len(df_final)-len(df_model))/len(df_final))*100:.1f}%")
print("="*60 + "\n")


# ---------------------------------------------------------
# 5. ANALYSE (WIE GEHABT)
# ---------------------------------------------------------

# MODELL 1: ZINB
print("MODELL 1: ZINB (total_bids)")
# Formel
count_formula = (
    "total_bids ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(procurement_category) + C(year)"
)
y_count, X_count = patsy.dmatrices(count_formula, data=df_model, return_type='dataframe')
X_infl = df_model[['const']]

zinb_model_instance = ZeroInflatedNegativeBinomialP(
    endog=y_count, exog=X_count, exog_infl=X_infl, inflation='logit'
)

try:
    print("Berechne ZINB...")
    res_nm = zinb_model_instance.fit(maxiter=10000, method='nm', disp=0)
    zinb_result = zinb_model_instance.fit(maxiter=10000, method='bfgs', start_params=res_nm.params, disp=0)
    print(zinb_result.summary())
except Exception as e:
    print(f"Fehler ZINB: {e}")

# MODELL 2: GLM
print("\nMODELL 2: Fractional Logit (sme_share)")
# Hier filtern wir zusätzlich auf total_bids > 0
df_model_h2 = df_model.dropna(subset=['sme_share']).copy()
epsilon = 1e-6
df_model_h2['sme_share_safe'] = df_model_h2['sme_share'].clip(epsilon, 1-epsilon)

print(f"Stichprobe GLM (nur Ausschreibungen mit Geboten): {len(df_model_h2)}")

glm_formula = (
    "sme_share_safe ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(procurement_category) + C(year)"
)

try:
    glm_model = smf.glm(
        formula=glm_formula,
        data=df_model_h2,
        family=sm.families.Binomial(link=sm.families.links.logit())
    ).fit(cov_type='HC0')
    print(glm_model.summary())
except Exception as e:
    print(f"Fehler GLM: {e}")

print("\n--- Analyse beendet ---")