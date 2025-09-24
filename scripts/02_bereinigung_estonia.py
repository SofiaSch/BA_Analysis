import pandas as pd
import os
import time

# --- Konfiguration ---
country_name = 'Estonia'
print(f"--- Starte Datenbereinigung & Variablenerstellung für {country_name} ---")

# --- Pfade dynamisch erstellen ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Eingabedatei ist das Ergebnis aus dem ersten Skript
raw_file_path = os.path.join(script_dir, '..', 'results', f'{country_name.lower()}_all_tenders_raw.csv')
# Ausgabedatei für den finalen, analysebereiten Datensatz
final_output_file = os.path.join(script_dir, '..', 'results', f'{country_name.lower()}_analysis_ready.csv')

# --- Schritt 1: Lade die aufbereiteten Rohdaten ---
print(f"Lade Rohdaten aus: {raw_file_path}")
start_time = time.time()
df = pd.read_csv(raw_file_path)
print(f"-> Fertig in {time.time() - start_time:.2f} Sekunden. {df.shape[0]} Zeilen geladen.")
initial_rows = len(df) # Wir merken uns die ursprüngliche Zeilenzahl

# --- Schritt 2: Datentypen korrigieren ---
print("\nSchritt 2: Wandle Datumsspalten um...")
# Wandel die Datums-Spalten in das korrekte datetime-Format um.
# Fehlerhafte Einträge werden zu 'NaT' (Not a Time), was als fehlender Wert behandelt wird.
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
print("-> Datumsspalten erfolgreich umgewandelt.")

# --- Schritt 3: Fehlende und unlogische Daten filtern ---
print("\nSchritt 3: Filtere fehlende und unlogische Daten...")
# 3a: Entferne alle Zeilen, in denen das Start- oder Enddatum fehlt.
# Ohne diese können wir die Dauer nicht berechnen.
df.dropna(subset=['publication_date', 'end_date'], inplace=True)
print(f"-> {initial_rows - len(df)} Zeilen wegen fehlender Daten entfernt.")
current_rows = len(df)

# 3b: Entferne unlogische Einträge, bei denen das Enddatum vor dem Startdatum liegt.
df = df[df['end_date'] >= df['publication_date']].copy()
print(f"-> {current_rows - len(df)} Zeilen wegen unlogischer Daten (Ende vor Start) entfernt.")
current_rows = len(df)

# 3c (Optional, aber empfohlen): Filtere nur auf wettbewerbliche Verfahren, die für deine Analyse relevant sind.
competitive_methods = ['open', 'selective'] # Passe diese Liste bei Bedarf an
df = df[df['procurement_method'].isin(competitive_methods)].copy()
print(f"-> {current_rows - len(df)} Zeilen wegen nicht-wettbewerblicher Verfahren entfernt.")

# --- Schritt 4: Variablen berechnen (Operationalisierung) ---
print("\nSchritt 4: Berechne die Analysevariablen...")
# 4a: Berechne die Dauer in Tagen (unsere unabhängige Variable)
df['duration_days'] = (df['end_date'] - df['publication_date']).dt.days

# 4b: Finalisiere die abhängigen Variablen (Anzahl der Gebote)
# Fehlende Werte bei den Geboten füllen wir mit 0 auf (Annahme: Kein Eintrag = 0 Gebote)
df['total_bids'].fillna(0, inplace=True)
df['sme_bids'].fillna(0, inplace=True)
# In saubere Ganzzahlen umwandeln
df['total_bids'] = df['total_bids'].astype(int)
df['sme_bids'] = df['sme_bids'].astype(int)
print("-> 'duration_days', 'total_bids' und 'sme_bids' finalisiert.")

# --- Schritt 5: Finalen Datensatz speichern ---
df.to_csv(final_output_file, index=False, encoding='utf-8-sig')

print("\n--- Prozess abgeschlossen! ---")
print(f"Der finale, analysebereite Datensatz hat {df.shape[0]} Zeilen.")
print(f"Gespeichert unter: {final_output_file}")

# Zeige eine Vorschau und wichtige Kennzahlen des finalen Datensatzes
print("\nAnalyse des finalen Datensatzes:")
print(df[['duration_days', 'total_bids', 'sme_bids', 'tender_value']].describe())