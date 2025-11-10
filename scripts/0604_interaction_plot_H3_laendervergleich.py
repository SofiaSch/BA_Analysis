import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- KONFIGURATION ---
PATH_TO_RESULTS = '../results/'
COL_DAUER = 'duration_days'
COL_GEBOTE_GESAMT = 'total_bids'
FILTER_TAGE = 200
BINS = 15 # Etwas weniger Bins, da wir auf 3 Länder aufteilen
# ---------------------

# 1. Daten laden
try:
    df_de = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'germany_analysis_ready.csv'))
    df_de['Land'] = 'Deutschland'
    df_fr = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'france_analysis_ready.csv'))
    df_fr['Land'] = 'Frankreich'
    df_ee = pd.read_csv(os.path.join(PATH_TO_RESULTS, 'estonia_analysis_ready.csv'))
    df_ee['Land'] = 'Estland'
    df = pd.concat([df_de, df_fr, df_ee], ignore_index=True)
except FileNotFoundError:
    exit()

# 2. Filtern
df_plot = df[(df[COL_DAUER] > 0) & (df[COL_DAUER] <= FILTER_TAGE)].copy()

# 3. Grafik erstellen
custom_palette = {'Deutschland': '#1f77b4', 'Frankreich': '#ff7f0e', 'Estland': '#2ca02c'}

# Wir nutzen hier x_bins, um Durchschnittswerte pro Land pro Zeit-Bucket zu zeigen
g = sns.lmplot(data=df_plot, x=COL_DAUER, y=COL_GEBOTE_GESAMT, hue='Land',
               x_bins=BINS, # Fasst Daten in Punkten zusammen
               height=6, aspect=1.5,
               scatter_kws={'s': 30, 'alpha': 0.6}, # Punkte etwas kleiner
               ci=None,
               palette=custom_palette,
               legend_out=False) # Legende in den Plot, spart Platz

# 4. Anpassungen
plt.title('Ländervergleich: Einfluss der Dauer auf den Wettbewerb (H3)', fontsize=14)
plt.xlabel('Geplante Verfahrensdauer (Tage)', fontsize=12)
plt.ylabel('Durchschnittliche Anzahl Gebote', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(left=0)

plt.tight_layout()
plt.show()