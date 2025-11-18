import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
import patsy
import warnings

# Warnungen unterdrücken
warnings.simplefilter('ignore')

print("--- Erstelle Visualisierungen (Prediction Plots) - FINAL FIX ---")

# 1. DATEN LADEN
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', 'results')

try:
    df_de = pd.read_csv(os.path.join(base_path, 'germany_analysis_ready.csv'), low_memory=False)
    df_fr = pd.read_csv(os.path.join(base_path, 'france_analysis_ready.csv'), low_memory=False)
    df_ee = pd.read_csv(os.path.join(base_path, 'estonia_analysis_ready.csv'), low_memory=False)
except:
    print("Fehler beim Laden. Prüfe Pfade.")
    exit()

df_de['country'] = 'Germany'
df_fr['country'] = 'France'
df_ee['country'] = 'Estonia'
df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
reg_df = df_final.copy()

# Bereinigung
reg_df = reg_df[reg_df['total_bids'] < 100]
reg_df['duration_days'] = pd.to_numeric(reg_df['duration_days'], errors='coerce')
reg_df = reg_df[reg_df['duration_days'] > 0]
p99 = reg_df['duration_days'].quantile(0.99)
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99)

# Standardisierung
duration_mean = reg_df['duration_days_capped'].mean()
duration_std = reg_df['duration_days_capped'].std()
reg_df['z_duration'] = (reg_df['duration_days_capped'] - duration_mean) / duration_std

reg_df['tender_value'] = pd.to_numeric(reg_df['tender_value'], errors='coerce')
reg_df = reg_df[reg_df['tender_value'] > 0]
reg_df['log_tender_value'] = np.log(reg_df['tender_value'])
reg_df['z_value'] = (reg_df['log_tender_value'] - reg_df['log_tender_value'].mean()) / reg_df['log_tender_value'].std()

reg_df['sme_share'] = np.where(reg_df['total_bids'] > 0, reg_df['sme_bids'] / reg_df['total_bids'], np.nan)
reg_df['sme_share'] = reg_df['sme_share'].clip(0, 1)
reg_df['const'] = 1

# Output Ordner
output_dir = os.path.join(script_dir, '..', 'plots')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sns.set_theme(style="whitegrid")
colors = {'Estonia': '#1f77b4', 'France': '#ff7f0e', 'Germany': '#2ca02c'}

# =============================================================================
# PLOT 1: BOXPLOT DAUER
# =============================================================================
print("Erstelle Plot 1: Boxplot Dauer...")
plt.figure(figsize=(10, 6))
sns.boxplot(x='country', y='duration_days_capped', data=reg_df, palette=colors, order=['Estonia', 'France', 'Germany'])
plt.title('Verteilung der Verfahrensdauer nach Ländern', fontsize=14)
plt.ylabel('Dauer (Tage)', fontsize=12)
plt.xlabel('Land', fontsize=12)
plt.savefig(os.path.join(output_dir, '01_boxplot_dauer.png'), dpi=300)
plt.close()

# =============================================================================
# MODELLE FIT (Nötig für Vorhersage)
# =============================================================================
print("Berechne Modelle für Vorhersage (bitte warten)...")

# ZINB
zinb_vars = ['total_bids', 'z_duration', 'country', 'z_value', 'procurement_method', 'procurement_category', 'year',
             'const']
df_z1 = reg_df.dropna(subset=zinb_vars)
formula_zinb = "total_bids ~ z_duration * C(country, Treatment('Estonia')) + z_value + C(procurement_method) + C(procurement_category) + C(year)"

# WICHTIG: Wir speichern X_z, um später das "Design" wiederzuverwenden!
y_z, X_z = patsy.dmatrices(formula_zinb, df_z1, return_type='dataframe')

# Modell fitten (für Plot reicht NM oder BFGS ohne SEs)
try:
    model_zinb = ZeroInflatedNegativeBinomialP(endog=y_z, exog=X_z, exog_infl=df_z1[['const']], inflation='logit').fit(
        maxiter=1000, method='bfgs', disp=0)
except:
    model_zinb = ZeroInflatedNegativeBinomialP(endog=y_z, exog=X_z, exog_infl=df_z1[['const']], inflation='logit').fit(
        maxiter=2000, method='nm', disp=0)

# GLM
df_glm = reg_df.dropna(subset=['sme_share'] + zinb_vars).copy()
df_glm['sme_share_safe'] = df_glm['sme_share'].clip(1e-6, 1 - 1e-6)
formula_glm = "sme_share_safe ~ z_duration * C(country, Treatment('Estonia')) + z_value + C(procurement_method) + C(procurement_category) + C(year)"
# Auch hier GLM fitten
model_glm = smf.glm(formula=formula_glm, data=df_glm, family=sm.families.Binomial(link=sm.families.links.logit())).fit()


# SYNTHETISCHE DATEN ERSTELLEN
def create_pred_data(df_orig):
    z_range = np.linspace(df_orig['z_duration'].min(), df_orig['z_duration'].max(), 100)
    days_range = (z_range * duration_std) + duration_mean

    pred_list = []
    # Wir nutzen Modus (häufigster Wert) für kategoriale Variablen
    mode_year = df_orig['year'].mode()[0]
    mode_method = df_orig['procurement_method'].mode()[0]
    mode_cat = df_orig['procurement_category'].mode()[0]
    mean_val = df_orig['z_value'].mean()

    for country in ['Estonia', 'France', 'Germany']:
        temp = pd.DataFrame({
            'z_duration': z_range,
            'days': days_range,
            'country': country,
            'z_value': mean_val,
            'const': 1,
            'year': mode_year,
            'procurement_method': mode_method,
            'procurement_category': mode_cat
        })
        pred_list.append(temp)
    return pd.concat(pred_list, ignore_index=True)


df_pred = create_pred_data(reg_df)

# =============================================================================
# PLOT 2: WETTBEWERB (ZINB) - MIT KORREKTUR
# =============================================================================
print("Erstelle Plot 2: Wettbewerbsintensität...")

# FIX: Wir nutzen build_design_matrices mit dem Bauplan (design_info) aus dem Training (X_z)
# Das stellt sicher, dass ALLE Spalten (auch die für andere Jahre) da sind, selbst wenn df_pred nur ein Jahr enthält.
X_pred_z = patsy.build_design_matrices([X_z.design_info], df_pred, return_type='dataframe')[0]

# Vorhersage
predicted_counts = model_zinb.predict(exog=X_pred_z, exog_infl=df_pred[['const']], which='mean')
df_pred['pred_bids'] = predicted_counts

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_pred, x='days', y='pred_bids', hue='country', palette=colors, linewidth=2.5)
plt.title('Vorhersage: Anzahl Gebote vs. Verfahrensdauer', fontsize=14)
plt.xlabel('Geplante Verfahrensdauer (Tage)', fontsize=12)
plt.ylabel('Vorhergesagte Anzahl Gebote', fontsize=12)
plt.legend(title='Land')
plt.ylim(0, 8)  # Zoom für bessere Sichtbarkeit
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_interaction_wettbewerb_H1_H3a.png'), dpi=300)
plt.close()

# =============================================================================
# PLOT 3: KMU (GLM)
# =============================================================================
print("Erstelle Plot 3: KMU-Anteil...")

# GLM predict via formula api ist intelligenter und braucht den Fix meist nicht,
# aber falls doch, macht statsmodels das intern.
pred_probs = model_glm.predict(df_pred)
df_pred['pred_sme'] = pred_probs

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_pred, x='days', y='pred_sme', hue='country', palette=colors, linewidth=2.5)
plt.title('Vorhersage: KMU-Anteil vs. Verfahrensdauer', fontsize=14)
plt.xlabel('Geplante Verfahrensdauer (Tage)', fontsize=12)
plt.ylabel('Vorhergesagter KMU-Anteil (0-1)', fontsize=12)
plt.legend(title='Land')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_interaction_kmu_H2_H3b.png'), dpi=300)
plt.close()

print(f"Fertig! Grafiken gespeichert in: {output_dir}")