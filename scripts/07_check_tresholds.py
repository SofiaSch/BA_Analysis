import pandas as pd
import os
import numpy as np

print("--- Prüfung der Auftragswerte (Schwellenwerte) ---")

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', 'results')

# Daten laden
try:
    df_de = pd.read_csv(os.path.join(base_path, 'germany_analysis_ready.csv'), low_memory=False)
    df_fr = pd.read_csv(os.path.join(base_path, 'france_analysis_ready.csv'), low_memory=False)
    df_ee = pd.read_csv(os.path.join(base_path, 'estonia_analysis_ready.csv'), low_memory=False)
except:
    print("Fehler: Daten nicht gefunden.")
    exit()

# Zusammenfügen
df_final = pd.concat([df_de, df_fr, df_ee], ignore_index=True)

# Filtern auf valide Werte (> 0)
values = df_final['tender_value'].dropna()
values = values[values > 0]

# Berechnungen
median_val = values.median()
mean_val = values.mean()
min_val = values.min()
max_val = values.max()

# Anteile berechnen
# EU-Schwellenwert für Dienstleistungen/Lieferungen (Zentralregierung) ca. 140.000€, sonst ca. 215.000€
# Bauleistungen ca. 5.3 Mio €.
# Wir nehmen mal 215.000 € als grobe Grenze für "Oberschwellig" bei Services.

count_under_25k = len(values[values < 25000])
count_under_100k = len(values[values < 100000])
count_under_215k = len(values[values < 215000])
total = len(values)

print(f"Anzahl Beobachtungen mit Wert: {total}")
print("-" * 40)
print(f"Minimum:    {min_val:,.2f} €")
print(f"Median:     {median_val:,.2f} €  <-- DAS IST DER ENTSCHEIDENDE WERT")
print(f"Mittelwert: {mean_val:,.2f} €")
print(f"Maximum:    {max_val:,.2f} €")
print("-" * 40)
print(f"Anteil unter 25.000 €:  {count_under_25k} ({count_under_25k/total:.1%})")
print(f"Anteil unter 100.000 €: {count_under_100k} ({count_under_100k/total:.1%})")
print(f"Anteil unter 215.000 €: {count_under_215k} ({count_under_215k/total:.1%})")
print("-" * 40)

if median_val > 200000:
    print("FAZIT: Eindeutig Oberschwellige (EU) Daten.")
elif median_val < 50000:
    print("FAZIT: Eindeutig viele Unterschwellige (nationale) Daten dabei.")
else:
    print("FAZIT: Mischmasch / Grauzone.")