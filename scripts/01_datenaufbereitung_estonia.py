import pandas as pd
import json
import os
import time

# --- Konfiguration ---
country_name = 'Estonia'
print(f"--- Starte Datenaufbereitung für {country_name} ---")

# --- Pfade dynamisch erstellen ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', country_name)
output_file = os.path.join(script_dir, '..', 'results', f'{country_name.lower()}_all_tenders_raw.csv')


# --- Hilfsfunktion zum Extrahieren der Daten aus einem JSON-Objekt ---
def extract_tender_data(tender_json, year):
    """
    Extrahiert alle für die Analyse benötigten Felder aus einem JSON-Objekt.
    Gibt ein "flaches" Dictionary zurück.
    """
    tender_info = tender_json.get('tender', {})

    # Extrahiere alle benötigten Felder sicher mit .get()
    publication_date = tender_json.get('date')
    end_date = tender_info.get('tenderPeriod', {}).get('endDate')

    tender_value = tender_info.get('value', {}).get('amount')
    procurement_method = tender_info.get('procurementMethod')
    procurement_category = tender_info.get('mainProcurementCategory')
    award_criteria = tender_info.get('awardCriteria')  # NEU
    tender_id = tender_info.get('id')

    # Extrahiere Gebotsstatistiken
    total_bids, sme_bids = None, None
    if 'statistics' in tender_json.get('bids', {}):
        for stat in tender_json['bids']['statistics']:
            if stat.get('measure') == 'electronicBids':
                total_bids = stat.get('value')
            if stat.get('measure') == 'smeBids':
                sme_bids = stat.get('value')

    return {
        'tender_id': tender_id,
        'publication_date': publication_date,
        'end_date': end_date,
        'total_bids': total_bids,
        'sme_bids': sme_bids,
        'tender_value': tender_value,
        'procurement_method': procurement_method,
        'procurement_category': procurement_category,
        'award_criteria': award_criteria,  # NEU
        'year': year  # NEU
    }


# --- Hauptverarbeitung ---
all_tenders_list = []
start_time = time.time()

for filename in os.listdir(data_path):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(data_path, filename)
        year = int(filename.split('_')[-1].replace('.jsonl', ''))  # Extrahiere das Jahr
        print(f"Verarbeite Datei: {filename}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tender_data_json = json.loads(line)
                extracted_data = extract_tender_data(tender_data_json, year)  # Übergebe das Jahr
                all_tenders_list.append(extracted_data)

# --- DataFrame erstellen und speichern ---
print("\nErstelle den finalen DataFrame...")
df_final = pd.DataFrame(all_tenders_list)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

end_time = time.time()
print(f"\n--- Verarbeitung für {country_name} abgeschlossen! ---")
print(f"Dauer: {end_time - start_time:.2f} Sekunden.")
print(f"Der finale Datensatz hat {df_final.shape[0]} Zeilen und {df_final.shape[1]} Spalten.")
print(f"Gespeichert unter: {output_file}")