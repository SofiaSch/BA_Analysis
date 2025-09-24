import pandas as pd
import os
import statsmodels.formula.api as smf

print("--- Starte finale Robustheitsanalysen mit OLS-Modell ---")

# ===================================================================
# SCHRITT 1: LADE UND FÜGE DIE BEREINIGTEN LÄNDERDATEN ZUSAMMEN
# ===================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# Pfade zu den bereinigten Dateien aus Schritt 2
germany_path = os.path.join(script_dir, '..', 'results', 'germany_analysis_ready.csv')
france_path = os.path.join(script_dir, '..', 'results', 'france_analysis_ready.csv')
estonia_path = os.path.join(script_dir, '..', 'results', 'estonia_analysis_ready.csv')

# Lade die drei DataFrames
df_de = pd.read_csv(germany_path)
df_fr = pd.read_csv(france_path)
df_ee = pd.read_csv(estonia_path)

# Füge eine Spalte für das Land hinzu (wichtig für die Analyse von H3!)
df_de['country'] = 'Germany'
df_fr['country'] = 'France'
df_ee['country'] = 'Estonia'

# Kombiniere die drei DataFrames zu einem Master-Datensatz
df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
print(f"Master-Datensatz mit {df_final.shape[0]} Zeilen erstellt.")


# ===================================================================
# SCHRITT 2: DATEN FÜR DIE REGRESSION VORBEREITEN (AUSREISSERBEHANDLUNG)
# ===================================================================
reg_df = df_final.copy()
reg_df = reg_df[reg_df['total_bids'] < 999] # Platzhalter entfernen
p99 = reg_df['duration_days'].quantile(0.99)
reg_df['duration_days_capped'] = reg_df['duration_days'].clip(upper=p99)
# Stelle sicher, dass alle für die Modelle benötigten Spalten keine fehlenden Werte haben
reg_df.dropna(subset=['total_bids', 'duration_days_capped', 'country', 'year', 'procurement_method', 'award_criteria'], inplace=True)
print(f"Finale Stichprobengröße für Analysen: {len(reg_df)} Zeilen.")


# ===================================================================
# ROBUSTHEITSANALYSE 1: ZUSÄTZLICHE KONTROLLVARIABLE 'AWARD_CRITERIA'
# ===================================================================
print("\n--- Robustheitsanalyse 1 (OLS): Zusätzliche Kontrollvariable 'award_criteria' ---")
ols_formula_robust = "total_bids ~ duration_days_capped * C(country) + C(year) + C(procurement_method) + C(award_criteria)"
model_ols_robust = smf.ols(ols_formula_robust, data=reg_df).fit()
print(model_ols_robust.summary())


# ===================================================================
# ROBUSTHEITSANALYSE 2: UNTERGRUPPE 'SERVICES'
# ===================================================================
print("\n--- Robustheitsanalyse 2 (OLS): Nur Ausschreibungen für 'services' ---")
reg_df_services = reg_df[reg_df['procurement_category'] == 'services'].copy()
print(f"Stichprobengröße für 'services': {len(reg_df_services)} Zeilen.")
if len(reg_df_services) > 0:
    base_formula_h3_ols = "total_bids ~ duration_days_capped * C(country) + C(year) + C(procurement_method)"
    model_ols_services = smf.ols(base_formula_h3_ols, data=reg_df_services).fit()
    print(model_ols_services.summary())


# ===================================================================
# ROBUSTHEITSANALYSE 3: AUSSCHLUSS DES STANDARD-ZEITRAUMS (30-35 TAGE)
# ===================================================================
print("\n--- Robustheitsanalyse 3 (OLS): Ausschluss der Standard-Dauer (30-35 Tage) ---")
reg_df_filtered_duration = reg_df[(reg_df['duration_days_capped'] < 30) | (reg_df['duration_days_capped'] > 35)].copy()
print(f"Stichprobengröße nach Filterung: {len(reg_df_filtered_duration)} Zeilen.")
if len(reg_df_filtered_duration) > 0:
    base_formula_h3_ols = "total_bids ~ duration_days_capped * C(country) + C(year) + C(procurement_method)"
    model_ols_filtered_duration = smf.ols(base_formula_h3_ols, data=reg_df_filtered_duration).fit()
    print(model_ols_filtered_duration.summary())