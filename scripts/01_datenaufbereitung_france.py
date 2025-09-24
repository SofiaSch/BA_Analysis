import pandas as pd
import os

# --- Dynamischen Pfad zum Datenordner erstellen ---

# 1. Finde heraus, wo das Skript selbst liegt
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Das Skript liegt im Ordner: {script_dir}")

# 2. Definiere das Land, das wir verarbeiten wollen
country_name = 'France'

# 3. Baue den Pfad zum richtigen Datenordner
# '..' bedeutet "gehe eine Ordnerebene nach oben" (von 'scripts' zu 'Analyse')
data_path = os.path.join(script_dir, '..', 'data', country_name)
output_file = os.path.join(script_dir, '..', 'data', f'{country_name.lower()}_all_years.csv')

print(f"Suche nach Daten im Ordner: {data_path}")

# --- Ab hier geht dein alter Code weiter ---

# Leere Liste, um die einzelnen Jahres-DataFrames zu sammeln
country_dataframes = []

# Durch alle Dateien im Ordner iterieren
print(f"Starte Verarbeitung für {country_name}...")

# WICHTIG: Prüfen, ob der Pfad überhaupt existiert, bevor wir weitermachen
if not os.path.isdir(data_path):
    print(f"FEHLER: Das Verzeichnis '{data_path}' wurde nicht gefunden. Bitte Pfad prüfen!")
else:
    for filename in os.listdir(data_path):
        # Nur Excel-Dateien berücksichtigen
        if filename.endswith('.xlsx'):
            # ... (der Rest deines Skripts bleibt gleich) ...
            print(f"--> Lese Datei: {filename}")

            file_path = os.path.join(data_path, filename)
            year = filename.split('_')[-1].replace('.xlsx', '')
            df = pd.read_excel(file_path)
            df['year'] = int(year)
            country_dataframes.append(df)

    if country_dataframes:
        full_country_df = pd.concat(country_dataframes, ignore_index=True)
        full_country_df.to_csv(output_file, index=False)
        print(f"\nVerarbeitung für {country_name} abgeschlossen!")
        print(f"Gespeicherte Datei: {output_file}")
        print(f"Der Datensatz hat {full_country_df.shape[0]} Zeilen und {full_country_df.shape[1]} Spalten.")
    else:
        print(f"Keine Excel-Dateien im Ordner {data_path} gefunden.")