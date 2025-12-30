# Datenanalyse: Verfahrensdauer und KMU-Beteiligung

Dieses Repository enthält die vollständige Pipeline zur Datenaufbereitung, Bereinigung und statistischen Analyse für die Bachelorarbeit:
**"Verfahrensdauer und Wettbewerb in der öffentlichen Auftragsvergabe"** (Sofia Schepers, TUM).

## 📁 Projektstruktur

Das Projekt ist in funktionale Module unterteilt, um eine klare Trennung zwischen Rohdaten, Prozessierung und Ergebnissen zu gewährleisten:

* `data/`: Enthält die länderspezifischen Rohdaten aus OpenTender.eu (Estland, Frankreich, Deutschland).
* `scripts/`: Der Kern der Analyse, unterteilt in:
    * `01_datenaufbereitung_*.py`: Extraktion der relevanten Variablen aus den JSON/CSV-Rohdaten.
    * `02_bereinigung_*.py`: Filterung von Ausreißern, Behandlung fehlender Werte und Logik-Checks.
    * `03_analyse_zinb_glm.py`: Implementierung der Hauptmodelle (Zero-Inflated Negative Binomial für Gebotsanzahl & Fractional Logit für KMU-Anteil).
    * `04_robustheitsanalyse.py`: Durchführung von Sensitivitätschecks (Dienstleistungssektor, alternative Modelle).
    * `05_zaehle_eintraege.py`: Generierung der Statistiken zur Stichprobenreduktion.
    * `06_visualisierung.py`: Erstellung der Interaktions-Plots und deskriptiven Grafiken.
    * `07_check_thresholds.py`: Validierung der Perzentil-Grenzwerte für die Hypothesentests.
* `results/`: Speichert die finalen bereinigten Datensätze (`analysis_ready.csv`) und tabellarischen Ergebnisse.
* `plots/`: Enthält die für die Thesis generierten Abbildungen (Boxplots, Regressionskurven).

## 🚀 Installation & Nutzung

### Voraussetzungen
* Python 3.12+
* Empfohlen: Nutzung einer virtuellen Umgebung (`venv` oder `conda`)

### Setup
1.  Klonen Sie das Repository:
    ```bash
    git clone [https://github.com/SofiaSch/BA_Analysis.git](https://github.com/SofiaSch/BA_Analysis.git)
    cd BA_Analysis
    ```
2.  Installieren Sie die benötigten Bibliotheken:
    ```bash
    pip install -r requirements.txt
    ```

### Analyse ausführen
Die Skripte sind nummeriert und sollten in der entsprechenden Reihenfolge ausgeführt werden, um die Datenpipeline korrekt zu durchlaufen (01 -> 02 -> 03).

## 📊 Methodik & Modelle

Die statistische Auswertung basiert auf zwei Hauptansätzen:
1.  **Wettbewerbsintensität:** Modellierung mittels **ZINB (Zero-Inflated Negative Binomial)**, um die hohe Anzahl an Nullgeboten (Zero-Inflation) und die Varianz der Gebote (Überdispersion) zu berücksichtigen.
2.  **KMU-Beteiligung:** Analyse des proportionalen KMU-Anteils über ein **Fractional Logit Modell** (GLM mit Binomial-Verteilung).

## 📝 Datenquelle
Die zugrunde liegenden Daten stammen von [OpenTender.eu](https://data.open-contracting.org/en/search/) und umfassen öffentliche Bekanntmachungen aus dem Zeitraum 2014–2022.

## ⚖️ Lizenz und Replizierbarkeit
Dieses Repository dient der wissenschaftlichen Transparenz. Der Code ist so dokumentiert, dass die in der Bachelorarbeit präsentierten Ergebnisse (Kapitel 5) eins-zu-eins repliziert werden können.