import pandas as pd

# Lade den konsolidierten Datensatz für Deutschland
df = pd.read_csv('Dokumente/Analyse/data/germany_all_years.csv')

# Zeige die ersten 5 Zeilen an, um einen Eindruck zu bekommen
print("Erste 5 Zeilen des Datensatzes:")
print(df.head())

# Zeige eine technische Zusammenfassung: Spalten, Anzahl der Einträge, Datentypen
print("\nÜbersicht der Spalten und Datentypen (df.info()):")
df.info()