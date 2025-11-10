import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION (BITTE ANPASSEN) ---
# Pfad zum Ordner 'results' relativ zum Speicherort dieses Skripts
PATH_TO_RESULTS = '../results/'

# Wie heißen deine Spalten exakt in den CSV-Dateien?
COL_DAUER = 'duration_days'  # z.B. 'duration_days', 'verfahrensdauer' etc.
# --------------------------------------

# --- 1. Daten laden und zusammenfügen ---
print("Lade Daten...")
try:
    df_de = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'germany_analysis_ready.csv'))
    df_de['Land'] = 'Deutschland'

    df_fr = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'france_analysis_ready.csv'))
    df_fr['Land'] = 'Frankreich'

    df_ee = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'estonia_analysis_ready.csv'))
    df_ee['Land'] = 'Estland'

    df = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
    print(f"Daten geladen. Gesamtanzahl Zeilen: {len(df)}")
except FileNotFoundError as e:
    print(f"FEHLER: Datei nicht gefunden. Überprüfe den Pfad '{PATH_TO_RESULTS}'")
    exit()

# --- 2. Grafik erstellen (Boxplot) ---
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Land', y=COL_DAUER, palette='Set2')

# --- 3. Anpassungen ---
plt.title('Vergleich der geplanten Verfahrensdauer nach Ländern', fontsize=14)
plt.ylabel('Geplante Verfahrensdauer (Tage)', fontsize=12)
plt.xlabel('Land', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.ylim(-10, 150) # Zeigt nur den Bereich bis 400 Tage an
plt.show()
# plt.savefig(os.path.join(PATH_TO_RESULTS, 'Grafik1_Boxplot_Dauer.png'), dpi=300)