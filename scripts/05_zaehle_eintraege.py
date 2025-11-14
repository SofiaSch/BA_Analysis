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
# --- NEU: Dictionary für die Zählung der Null-Gebote ---
zero_bid_counts = {}

print("--- Beginne mit der Analyse der Dateien ---\n")

# Schleife durch die Länder, um jede Datei zu verarbeiten
for country, files in countries.items():
    try:
        # Pfad zur Rohdatendatek erstellen
        raw_path = os.path.join(base_path, files['raw'])
        # CSV einlesen und Zeilen zählen
        raw_df = pd.read_csv(raw_path, low_memory=False)  # low_memory=False hinzugefügt, um DtypeWarning zu vermeiden
        raw_counts[country] = len(raw_df)

        # Pfad zur bereinigten Datei erstellen
        ready_path = os.path.join(base_path, files['ready'])
        # CSV einlesen und Zeilen zählen
        ready_df = pd.read_csv(ready_path, low_memory=False)
        ready_counts[country] = len(ready_df)

        # --- NEU: Zähle Nullen für H1-Diagnose ---
        # Überprüfe, ob die Spalte 'total_bids' existiert
        if 'total_bids' in ready_df.columns:
            # Zähle, wie oft total_bids == 0 ist
            zeros = (ready_df['total_bids'] == 0).sum()
            zero_bid_counts[country] = zeros
        else:
            print(f"WARNUNG: Spalte 'total_bids' nicht in {files['ready']} gefunden.")
            zero_bid_counts[country] = 0
        # --- ENDE NEU ---

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

    # --- NEU: Ausgabe der Zero-Inflation-Diagnose ---
    if zero_bid_counts:
        print("--- Diagnose: Anteil der Null-Gebote (total_bids == 0) in bereinigten Daten ---")

        # Erstelle ein DataFrame für die Zusammenfassung
        df_zeros_summary = pd.DataFrame({
            'Gesamt (bereinigt)': ready_counts,
            'davon Null-Gebote': zero_bid_counts
        }).reindex(countries.keys())  # Stellt sicher, dass die Reihenfolge DE, FR, EE ist

        df_zeros_summary['Anteil Nullen (%)'] = (df_zeros_summary['davon Null-Gebote'] / df_zeros_summary[
            'Gesamt (bereinigt)']) * 100

        # Gib die Tabelle im Markdown-Format aus (sehr sauber in der Konsole)
        print(df_zeros_summary.to_markdown(floatfmt=",.2f"))

        # Gesamtergebnis
        total_zeros = sum(zero_bid_counts.values())
        percent_total_zeros = (total_zeros / total_ready) * 100

        print("\n----------------------------------------------------")
        print(f"Gesamte Null-Gebote:   {total_zeros:d} (von {total_ready:d} Einträgen)")
        print(f"Gesamtanteil Nullen:   {percent_total_zeros:.2f}%")
        print("----------------------------------------------------")

        if percent_total_zeros > 15:
            print("\nEMPFEHLUNG: Der Anteil an Nullen ist hoch (>15%).")
            print("Ein Zero-Inflated Negative Binomial (ZINB) Modell wird für H1 dringend empfohlen.")
        else:
            print("\nEMPFEHLUNG: Der Anteil an Nullen ist moderat.")
            print(
                "Eine Negative Binomialregression (NBR) ist wahrscheinlich ausreichend, aber ZINB bleibt eine Option.")

    # Den Zielsatz mit den neu berechneten Werten erstellen
    print("\n--- Korrigierter Satz für deine Bachelorarbeit (Kapitel 4.1.3) ---")
    final_sentence = (
        f"Durch die Implementierung dieser Schritte wurde eine Reduktion der Datenmenge von {total_raw:,.0f} "
        f"Einträgen (Deutschland: {raw_counts.get('Deutschland', 0):,.0f}, Frankreich: {raw_counts.get('Frankreich', 0):,.0f}, "
        f"Estland: {raw_counts.get('Estland', 0):,.0f}) auf {total_ready:,.0f} Einträgen "
        f"(Deutschland: {ready_counts.get('Deutschland', 0):,.0f}, Frankreich: {ready_counts.get('Frankreich', 0):,.0f}, "
        f"Estland: {ready_counts.get('Estland', 0):,.0f}) durchgeführt, um eine "
        "konsistente und signifikante Stichprobe zu erhalten."
    )
    print(final_sentence)

else:
    print("Es konnten keine Daten gezählt werden. Bitte überprüfe die Fehlermeldungen oben.")