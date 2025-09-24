import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# ===================================================================
# ZUERST: Daten laden und zusammenfügen
# ===================================================================

# --- Pfade zu den bereinigten Dateien ---
script_dir = os.path.dirname(os.path.abspath(__file__))
germany_path = os.path.join(script_dir, '..', 'results', 'germany_analysis_ready.csv')
france_path = os.path.join(script_dir, '..', 'results', 'france_analysis_ready.csv')
estonia_path = os.path.join(script_dir, '..', 'results', 'estonia_analysis_ready.csv')

# --- Lade die drei DataFrames ---
df_de = pd.read_csv(germany_path)
df_fr = pd.read_csv(france_path)
df_ee = pd.read_csv(estonia_path)

# --- Füge eine Spalte für das Land hinzu ---
df_de['country'] = 'Germany'
df_fr['country'] = 'France'
df_ee['country'] = 'Estonia'

# --- Kombiniere die drei DataFrames zu 'df_final' ---
df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)

print("Alle drei Länder erfolgreich zusammengeführt!")
print(f"Der finale Master-Datensatz hat {df_final.shape[0]} Zeilen.")

# ===================================================================
# NEU: Daten für die Regression vorbereiten (Ausreißer behandeln)
# ===================================================================
print("\n--- Bereite Daten für die Regression vor (Ausreißerbehandlung) ---")

# Wir erstellen eine Kopie, um die Originaldaten nicht zu verändern
reg_df = df_final.copy()

# 1. Behandle Ausreißer in der abhängigen Variable (total_bids)
# Der Wert 999 ist unrealistisch. Wir entfernen diese Zeilen.
initial_rows = len(reg_df)
reg_df = reg_df[reg_df['total_bids'] < 999]
print(f"-> {initial_rows - len(reg_df)} Zeilen mit total_bids >= 999 entfernt.")

# 2. Behandle Ausreißer in der unabhängigen Variable (duration_days)
# Wir kappen (winsorizen) die Dauer beim 99. Perzentil.
# Alle Werte darüber werden auf diesen Maximalwert gesetzt.
p99 = reg_df['duration_days'].quantile(0.99)
print(f"-> Dauer wird beim 99. Perzentil gekappt: {p99:.0f} Tage.")
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99)

# 3. Entferne Zeilen mit fehlenden Werten in den verwendeten Variablen
reg_df.dropna(subset=['total_bids', 'duration_days_capped', 'country', 'year', 'procurement_method'], inplace=True)
print(f"-> Finale Stichprobengröße für Regression: {len(reg_df)} Zeilen.")

# ===================================================================
# Regressionen auf den bereinigten Daten erneut ausführen
# ===================================================================

# --- Hypothesentest für H1 ---
print("\n--- Starte Regression für H1 (bereinigte Daten) ---")
# Wir benutzen jetzt unsere neue, gekappte Variable 'duration_days_capped'
model_formula_h1 = "total_bids ~ duration_days_capped + C(country) + C(year) + C(procurement_method)"
model_h1 = smf.negativebinomial(model_formula_h1, data=reg_df).fit(disp=False)
print(model_h1.summary())

# --- Hypothesentest für H3 ---
print("\n--- Starte Regressionsmodell für H3 (bereinigte Daten) ---")
model_formula_h3 = "total_bids ~ duration_days_capped * C(country) + C(year) + C(procurement_method)"
model_h3 = smf.negativebinomial(model_formula_h3, data=reg_df).fit(disp=False)
print(model_h3.summary())