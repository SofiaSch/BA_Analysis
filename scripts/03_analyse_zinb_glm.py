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

print("--- Starte ZINB & GLM Analyse (inkl. Deskriptiver Statistik für BA) ---")

# Pfade setzen
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', 'results')

# ---------------------------------------------------------
# 1. DATEN LADEN & URSPRUNGS-ZÄHLUNG
# ---------------------------------------------------------
print("Lade Daten...")
try:
    # Hinweis: Passe die Dateinamen ggf. an, falls sie bei dir anders heißen
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
# Wir nutzen für die Statistik die gecappten Werte, um Ausreißer nicht zu stark zu gewichten,
# aber für das Verständnis sind "echte Tage" wichtig.
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99_dur)

# Z-Standardisierung für Regression
reg_df['z_duration'] = (
    reg_df['duration_days_capped'] - reg_df['duration_days_capped'].mean()
) / reg_df['duration_days_capped'].std()

# C. Tender Value
reg_df['tender_value'] = pd.to_numeric(reg_df['tender_value'], errors='coerce')
reg_df = reg_df[reg_df['tender_value'] > 0]
reg_df['log_tender_value'] = np.log(reg_df['tender_value'])
reg_df['z_value'] = (
    reg_df['log_tender_value'] - reg_df['log_tender_value'].mean()
) / reg_df['log_tender_value'].std()

# D. H2 Variable (KMU Anteil)
# Logik: Wenn Bids > 0, berechne Anteil. Wenn Bids = 0, ist Anteil NaN (nicht 0!)
reg_df['sme_share'] = np.where(
    reg_df['total_bids'] > 0,
    reg_df['sme_bids'] / reg_df['total_bids'],
    np.nan
)
reg_df['sme_share'] = reg_df['sme_share'].clip(0, 1)

# E. Konstante für Regression
reg_df['const'] = 1

# ---------------------------------------------------------
# 3. FINALER STICHPROBEN-FILTER (LISTWISE DELETION)
# ---------------------------------------------------------
# Variablen für Modell 1
model_vars = ['total_bids', 'z_duration', 'country', 'z_value',
              'procurement_method', 'procurement_category', 'year', 'const']

# Finaler Datensatz
df_model = reg_df.dropna(subset=model_vars).copy()

# Zählung pro Land im finalen Datensatz
counts_final = df_model['country'].value_counts()

# ---------------------------------------------------------
# 4. AUSGABE DER STICHPROBEN-STATISTIK (Tabelle 2)
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
# 4a. BERECHNUNG DER WERTE FÜR TABELLE 3 (ERGEBNISSE)
# ---------------------------------------------------------
print("\n" + "="*60)
print("WERTE FÜR TABELLE 3 (DESKRIPTIVE STATISTIKEN)")
print("="*60)

# Hilfsfunktion für Zero-Inflation (Anteil der Nullen in %)
def zero_share_pct(x):
    return (x == 0).mean() * 100

# Aggregation der Statistiken nach Land
# Wir nutzen duration_days_capped für die Statistik, da dies robuster ist
desc_stats = df_model.groupby('country').agg({
    'duration_days_capped': ['mean', 'std', 'median'],
    'total_bids': ['mean', 'std', zero_share_pct],
    'sme_share': ['mean', 'std'], # Ignoriert automatisch NaNs (also Fälle mit 0 Geboten)
    'tender_value': ['median']
})

# Umbenennen für Lesbarkeit
desc_stats.columns = [
    'Dauer_Mittelwert', 'Dauer_SD', 'Dauer_Median',
    'Gebote_Mittelwert', 'Gebote_SD', 'Null_Gebote_Pct',
    'KMU_Anteil_Mittelwert', 'KMU_Anteil_SD',
    'Wert_Median'
]

# Transponieren für bessere Lesbarkeit im Terminal
print(desc_stats.round(2).T)
print("-" * 60)
print("HINWEIS: Übertrage diese Werte in deine LaTeX Tabelle 3.")
print("="*60 + "\n")


# ---------------------------------------------------------
# 5. ANALYSE (REGRESSIONEN FÜR TABELLE 4 & 5)
# ---------------------------------------------------------

# MODELL 1: ZINB (H1 & H3 - Wettbewerb)
print("MODELL 1: ZINB (total_bids)")
# Referenzkategorie Estland, Interaktionen für DE und FR
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
    print("Berechne ZINB (kann kurz dauern)...")
    res_nm = zinb_model_instance.fit(maxiter=10000, method='nm', disp=0)
    zinb_result = zinb_model_instance.fit(maxiter=10000, method='bfgs', start_params=res_nm.params, disp=0)
    print(zinb_result.summary())
except Exception as e:
    print(f"Fehler ZINB: {e}")

# MODELL 2: GLM (H2 & H3 - KMU Anteil)
print("\nMODELL 2: Fractional Logit (sme_share)")
# Filtern auf erfolgreiche Ausschreibungen (Gebote > 0)
df_model_h2 = df_model.dropna(subset=['sme_share']).copy()

# Epsilon-Korrektur für Fractional Logit (0/1 Ränder)
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

# ---------------------------------------------------------
# 6. BERECHNUNG DER KORRELATIONSMATRIX (Für Anhang A.2)
# ---------------------------------------------------------
print("\n" + "="*60)
print("LATEX CODE FÜR ANHANG A.2 (KORRELATIONSMATRIX)")
print("="*60)

# Variablen für die Matrix auswählen (nur numerische Kernvariablen)
corr_vars = ['total_bids', 'sme_share', 'z_duration', 'z_value']
# Wir nutzen df_model, aber SME-Share ist dort oft NaN (bei 0 Geboten).
# Für die Korrelation nutzen wir alle verfügbaren Paare (pairwise deletion).
corr_matrix = df_model[corr_vars].corr()

# Vorbereitung für APA-Stil (untere Dreiecksmatrix)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
apa_corr = corr_matrix.mask(mask)

# Erstellung des DataFrames für den Export
var_labels = [
    "1. Gebotsanzahl (total_bids)",
    "2. KMU-Anteil (sme_share)",
    "3. Verf.-dauer (z_duration)",
    "4. Auftragswert (z_value)"
]
final_corr_df = pd.DataFrame(index=var_labels)

# Mittelwerte und Standardabweichungen (SD) hinzufügen
# Hinweis: M/SD für sme_share wird nur über Fälle mit Geboten berechnet
final_corr_df['M'] = [df_model[v].mean() for v in corr_vars]
final_corr_df['SD'] = [df_model[v].std() for v in corr_vars]

# Korrelationsspalten 1 bis 4 hinzufügen
for i in range(len(corr_vars)):
    col_values = apa_corr.iloc[:, i].values
    formatted_col = [f"{v:.2f}" if pd.notnull(v) else ("-" if j==i else "")
                     for j, v in enumerate(col_values)]
    final_corr_df[str(i+1)] = formatted_col

# LaTeX Output drucken
print(final_corr_df.round(2).to_latex(
    column_format='l cc cccc',
    caption="Mittelwerte, Standardabweichungen und Korrelationen der Kernvariablen",
    label="tab:correlation_matrix"
))

print(f"Anmerkung: N = {len(df_model)} (sme_share basiert auf n = {len(df_model_h2)})")
print("="*60 + "\n")

print("\n--- Analyse beendet ---")