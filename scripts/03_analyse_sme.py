import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf

# --- Konfiguration ---
print("--- Starte Analyse für H2: KMU-Beteiligung ---")

# --- Schritt 1: Lade und kombiniere die drei Datensätze ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# (Der Code zum Laden und Zusammenfügen der 3 Länder-Dateien ist identisch zum vorherigen Skript)
germany_path = os.path.join(script_dir, '..', 'results', 'germany_analysis_ready.csv')
france_path = os.path.join(script_dir, '..', 'results', 'france_analysis_ready.csv')
estonia_path = os.path.join(script_dir, '..', 'results', 'estonia_analysis_ready.csv')

df_de = pd.read_csv(germany_path)
df_fr = pd.read_csv(france_path)
df_ee = pd.read_csv(estonia_path)

df_de['country'] = 'Germany'
df_fr['country'] = 'France'
df_ee['country'] = 'Estonia'

df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
print(f"Master-Datensatz mit {df_final.shape[0]} Zeilen geladen.")


# --- Schritt 2: Daten für die Regression vorbereiten (Ausreißer behandeln) ---
print("\n--- Bereite Daten für die Regression vor ---")
reg_df = df_final.copy()

# 2a: Behandle Ausreißer in sme_bids und duration_days
initial_rows = len(reg_df)
reg_df = reg_df[reg_df['sme_bids'] < 999] # Annahme, dass 999 ein Platzhalter ist
print(f"-> {initial_rows - len(reg_df)} Zeilen mit sme_bids >= 999 entfernt.")

p99 = reg_df['duration_days'].quantile(0.99)
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99)
print(f"-> Dauer wird beim 99. Perzentil gekappt: {p99:.0f} Tage.")

# 2b: Entferne Zeilen mit fehlenden Werten
reg_df.dropna(subset=['sme_bids', 'duration_days_capped', 'country', 'year', 'procurement_method'], inplace=True)
print(f"-> Finale Stichprobengröße für Regression: {len(reg_df)} Zeilen.")

# --- Schritt 3a: Test für H2 (Absolute Anzahl der KMU-Gebote) ---
print("\n--- Starte Regression für H2 (Absolute Anzahl KMU-Gebote) ---")
model_formula_h2 = "sme_bids ~ duration_days_capped + C(country) + C(year) + C(procurement_method)"
model_h2 = smf.negativebinomial(model_formula_h2, data=reg_df).fit(disp=False)
print(model_h2.summary())

# --- Schritt 3b: Test für H2 (Prozentualer Anteil der KMU-Gebote) ---
print("\n--- Starte Regression für H2 (Prozentualer Anteil KMU-Gebote) ---")

# Berechne den Anteil. Wichtig: total_bids darf nicht 0 sein, um Teilung durch Null zu vermeiden.
# Wo total_bids=0 ist, ist der Anteil auch 0.
reg_df['sme_share'] = np.where(reg_df['total_bids'] > 0, reg_df['sme_bids'] / reg_df['total_bids'], 0)

# Hier verwenden wir ein Standard OLS-Modell (lineare Regression), da der Anteil eine kontinuierliche Variable ist.
model_formula_h2_share = "sme_share ~ duration_days_capped + C(country) + C(year) + C(procurement_method)"
model_h2_share = smf.ols(model_formula_h2_share, data=reg_df).fit()
print(model_h2_share.summary())