import pandas as pd
import os

# --- KORREKTUR START ---
# Finde den absoluten Pfad des Verzeichnisses, in dem das Skript liegt (also .../scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Baue den korrekten Pfad zum 'results'-Ordner, indem wir eine Ebene nach oben gehen ('..')
# und dann in 'results' wechseln.
base_path = os.path.join(script_dir, '..', 'results')
# --- KORREKTUR ENDE ---


# Eine Liste der Länder und ihrer zugehörigen Dateinamen
countries = {
    'Deutschland': {
        'raw': 'germany_all_tenders_raw.csv',
        'ready': 'germany_analysis_ready.csv'
    },
    'Frankreich': {
        'raw': 'france_all_tenders_raw.csv',
        'ready': 'france_analysis_ready.csv'
    },
    'Estland': {
        'raw': 'estonia_all_tenders_raw.csv',
        'ready': 'estonia_analysis_ready.csv'
    }
}

# Dictionaries, um die Ergebnisse zu speichern
raw_counts = {}
ready_counts = {}

print("--- Beginne mit der Analyse der Dateien ---\n")

# Schleife durch die Länder, um jede Datei zu verarbeiten
for country, files in countries.items():
    try:
        # Pfad zur Rohdatendatek erstellen
        raw_path = os.path.join(base_path, files['raw'])
        # CSV einlesen und Zeilen zählen
        raw_df = pd.read_csv(raw_path)
        raw_counts[country] = len(raw_df)

        # Pfad zur bereinigten Datei erstellen
        ready_path = os.path.join(base_path, files['ready'])
        # CSV einlesen und Zeilen zählen
        ready_df = pd.read_csv(ready_path)
        ready_counts[country] = len(ready_df)

        print(f"Erfolgreich verarbeitet: {country}")

    except FileNotFoundError as e:
        print(f"FEHLER: Datei nicht gefunden - {e}. Bitte überprüfe den Pfad und Dateinamen.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist bei der Verarbeitung von {country} aufgetreten: {e}")

print("\n--- Analyse abgeschlossen ---\n")

# Überprüfen, ob Daten gesammelt wurden, bevor die Gesamtsummen berechnet werden
if raw_counts and ready_counts:
    # Gesamtzahlen berechnen
    total_raw = sum(raw_counts.values())
    total_ready = sum(ready_counts.values())

    # Ergebnisse formatiert ausgeben
    print("--- Ergebnisse der Rohdaten (Vor der Bereinigung) ---")
    print(f"Deutschland (roh): {raw_counts.get('Deutschland', 0):,}")
    print(f"Frankreich (roh):  {raw_counts.get('Frankreich', 0):,}")
    print(f"Estland (roh):     {raw_counts.get('Estland', 0):,}")
    print("----------------------------------------------------")
    print(f"Gesamteinträge (roh):      {total_raw:,}\n")

    print("--- Ergebnisse der bereinigten Daten (Nach der Bereinigung) ---")
    print(f"Deutschland (bereinigt): {ready_counts.get('Deutschland', 0):,}")
    print(f"Frankreich (bereinigt):  {ready_counts.get('Frankreich', 0):,}")
    print(f"Estland (bereinigt):     {ready_counts.get('Estland', 0):,}")
    print("----------------------------------------------------")
    print(f"Gesamteinträge (bereinigt):  {total_ready:,}\n")

    # Den Zielsatz mit den neu berechneten Werten erstellen
    print("--- Korrigierter Satz für deine Bachelorarbeit ---")
    final_sentence = (
        f"Durch diese Schritte haben wir die Datenmenge von {total_raw:,} Einträgen "
        f"(Deutschland: {raw_counts.get('Deutschland', 0):,}, Frankreich: {raw_counts.get('Frankreich', 0):,}, "
        f"Estland: {raw_counts.get('Estland', 0):,}) auf {total_ready:,} Einträgen "
        f"(Deutschland: {ready_counts.get('Deutschland', 0):,}, Frankreich: {ready_counts.get('Frankreich', 0):,}, "
        f"Estland: {ready_counts.get('Estland', 0):,}) reduziert, um eine saubere und "
        "relevante Stichprobe zu erhalten."
    )
    print(final_sentence)
else:
    print("Es konnten keine Daten gezählt werden. Bitte überprüfe die Fehlermeldungen oben.")