import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import patsy
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Warnungen unterdrücken für sauberen Output
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

print("--- Starte ZINB & GLM Analyse (Finaler Tutor-Code) ---")

# Pfade setzen
script_dir = os.path.dirname(os.path.abspath(__file__))
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

# 1. Extremwerte filtern (Technischer Schutz für ZINB)
reg_df = reg_df[reg_df['total_bids'] < 100]

# 2. Duration: Numerisch machen, Bereinigen, Cappen, Standardisieren
reg_df['duration_days'] = pd.to_numeric(reg_df['duration_days'], errors='coerce')
reg_df = reg_df[reg_df['duration_days'] > 0]  # Keine negativen Tage

# Capping bei 99% (gegen Ausreißer)
p99_dur = reg_df['duration_days'].quantile(0.99)
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99_dur)

# Z-Standardisierung
reg_df['z_duration'] = (
                               reg_df['duration_days_capped'] - reg_df['duration_days_capped'].mean()
                       ) / reg_df['duration_days_capped'].std()

# 3. Tender Value: Log & Standardisierung
reg_df['tender_value'] = pd.to_numeric(reg_df['tender_value'], errors='coerce')
reg_df = reg_df[reg_df['tender_value'] > 0]  # Nur positive Werte für Log
reg_df['log_tender_value'] = np.log(reg_df['tender_value'])
reg_df['z_value'] = (
                            reg_df['log_tender_value'] - reg_df['log_tender_value'].mean()
                    ) / reg_df['log_tender_value'].std()

# 4. H2 Variable: KMU Anteil (Fractional Response)
# Berechnung nur sinnvoll, wo es überhaupt Gebote gab
reg_df['sme_share'] = np.where(
    reg_df['total_bids'] > 0,
    reg_df['sme_bids'] / reg_df['total_bids'],
    np.nan  # NaN setzen, wird später gefiltert
)
reg_df['sme_share'] = reg_df['sme_share'].clip(0, 1)

# 5. Konstante für ZINB Inflation hinzufügen
reg_df['const'] = 1

# 6. Missing Values entfernen (für saubere Modellierung)
model_vars = ['total_bids', 'z_duration', 'country', 'z_value',
              'procurement_method', 'procurement_category', 'year', 'const', 'sme_share']
# Wir droppen NaNs erst in den spezifischen Modell-Subsets,
# da sme_share viele NaNs hat (wo total_bids=0)

print(f"Basis-Datensatzgröße nach Bereinigung: {len(reg_df)}")

# ---------------------------------------------------------
# 2. MODELL 1: ZINB (H1 & H3a - Wettbewerbsintensität)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MODELL 1: ZINB (total_bids)")
print("=" * 60)

# Subset für ZINB (alle Fälle, wo IVs da sind)
zinb_vars = ['total_bids', 'z_duration', 'country', 'z_value',
             'procurement_method', 'procurement_category', 'year', 'const']
reg_df_z1 = reg_df.dropna(subset=zinb_vars).copy()
print(f"Stichprobe ZINB: {len(reg_df_z1)}")

# Formel (Count-Teil): Estland ist Referenz durch Treatment('Estonia')
count_formula = (
    "total_bids ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(procurement_category) + C(year)"
)

# Design-Matrizen erstellen
y_count, X_count = patsy.dmatrices(count_formula, data=reg_df_z1, return_type='dataframe')
X_infl = reg_df_z1[['const']]  # Konstante Inflation

# Fitting versuchen (Nelder-Mead ist robuster bei Zero-Inflation)
try:
    print("Starte Optimierung (Methode: 'nm' - Nelder-Mead)...")
    zinb_model = ZeroInflatedNegativeBinomialP(
        endog=y_count,
        exog=X_count,
        exog_infl=X_infl,
        inflation='logit'
    ).fit(maxiter=15000, method='nm', disp=False)

    print(zinb_model.summary())

except Exception as e:
    print(f"Fehler bei ZINB (nm): {e}")
    print("Versuche Fallback auf 'bfgs'...")
    try:
        zinb_model = ZeroInflatedNegativeBinomialP(
            endog=y_count,
            exog=X_count,
            exog_infl=X_infl,
            inflation='logit'
        ).fit(maxiter=10000, method='bfgs', disp=False)
        print(zinb_model.summary())
    except Exception as e2:
        print(f"Modell konvergiert nicht: {e2}")

# ---------------------------------------------------------
# 3. MODELL 2: GLM Fractional Logit (H2 & H3b - KMU)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("MODELL 2: Fractional Logit (sme_share)")
print("=" * 60)

# Nur Fälle mit Geboten > 0 (Selektionseffekt vermeiden)
reg_df_h2 = reg_df.dropna(subset=['sme_share'] + zinb_vars).copy()

# FIX: Werte minimal von 0 und 1 wegschieben, um log(0) Crash zu verhindern
epsilon = 1e-6
reg_df_h2['sme_share_safe'] = reg_df_h2['sme_share'].clip(epsilon, 1 - epsilon)

print(f"Stichprobe GLM: {len(reg_df_h2)}")

glm_formula = (
    "sme_share_safe ~ z_duration * C(country, Treatment('Estonia')) + "
    "z_value + C(procurement_method) + C(procurement_category) + C(year)"
)

try:
    # Quasi-Maximum Likelihood Estimation (QMLE) via Binomial Family
    # cov_type='HC0' für robuste Standardfehler
    glm_model = smf.glm(
        formula=glm_formula,
        data=reg_df_h2,
        family=sm.families.Binomial(link=sm.families.links.logit())
    ).fit(cov_type='HC0')

    print(glm_model.summary())

except Exception as e:
    print(f"Fehler beim GLM-Fitting: {e}")

print("\n--- Analyse beendet ---")