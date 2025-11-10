import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
PATH_TO_RESULTS = '../results/'
COL_DAUER = 'duration_days'
COL_GEBOTE_KMU = 'sme_bids'
FILTER_TAGE = 365
BINS = 20
# ---------------------

# 1. Daten laden
try:
    df_de = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'germany_analysis_ready.csv'))
    df_fr = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'france_analysis_ready.csv'))
    df_ee = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'estonia_analysis_ready.csv'))
    df = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
except FileNotFoundError:
    exit()

# 2. Filtern
df_plot = df[(df[COL_DAUER] > 0) & (df[COL_DAUER] <= FILTER_TAGE)].copy()

# 3. Grafik erstellen
plt.figure(figsize=(10, 6))

sns.regplot(data=df_plot, x=COL_DAUER, y=COL_GEBOTE_KMU,
            x_bins=BINS,
            scatter_kws={'alpha': 0.6, 's': 60, 'color': 'forestgreen', 'edgecolor': 'k'},
            line_kws={'color': 'darkgreen', 'linewidth': 2.5})

# 4. Anpassungen
plt.title('H2: Einfluss der Verfahrensdauer auf die KMU-Beteiligung', fontsize=14)
plt.xlabel('Geplante Verfahrensdauer (Tage, gruppiert)', fontsize=12)
plt.ylabel('Durchschnittliche Anzahl KMU-Gebote', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()
# plt.savefig(os.path.join(PATH_TO_RESULTS, 'Grafik3_H2_KMU.png'), dpi=300)