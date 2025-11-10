import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
PATH_TO_RESULTS = '../results/'
COL_DAUER = 'duration_days'
COL_GEBOTE_GESAMT = 'total_bids'
FILTER_TAGE = 365 # Wir schauen uns das "normale" Jahr an, um Extreme auszuschließen
BINS = 20         # Fasst die x-Achse in 20 gleich große Abschnitte zusammen
# ---------------------

# 1. Daten laden
try:
    df_de = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'germany_analysis_ready.csv'))
    df_fr = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'france_analysis_ready.csv'))
    df_ee = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'estonia_analysis_ready.csv'))
    df = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
except FileNotFoundError:
    print("Daten nicht gefunden.")
    exit()

# 2. Filtern für saubere Darstellung
# Wir nehmen nur Verfahren > 0 Tage und <= 365 Tage
df_plot = df[(df[COL_DAUER] > 0) & (df[COL_DAUER] <= FILTER_TAGE)].copy()

# 3. Grafik erstellen (Binscatter)
plt.figure(figsize=(10, 6))

sns.regplot(data=df_plot, x=COL_DAUER, y=COL_GEBOTE_GESAMT,
            x_bins=BINS, # Der Trick für übersichtliche Grafiken bei vielen Daten
            scatter_kws={'alpha': 0.6, 's': 60, 'color': 'grey', 'edgecolor': 'k'},
            line_kws={'color': 'darkblue', 'linewidth': 2.5})

# 4. Anpassungen
plt.title('H1: Einfluss der Verfahrensdauer auf die Wettbewerbsintensität', fontsize=14)
plt.xlabel('Geplante Verfahrensdauer (Tage, gruppiert)', fontsize=12)
plt.ylabel('Durchschnittliche Anzahl aller Gebote', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(PATH_TO_RESULTS, 'Grafik2_H1_Wettbewerb.png'), dpi=300)