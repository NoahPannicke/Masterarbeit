# Masterarbeit
Enthält alle Skripte die zur Umsetzung der Masterarbeit mit dem Titel "Entwicklung eines Machine-Learning-Modells zur Prognose von Produktaffinität in einer gesetzlichen Krankenversicherung" verwendet wurden.

Alle Analysen sind in zwei zentralen Skripten zusammengefasst: train_test_models_ma.R und descriptive_analysis_ma.R.
Des Weiteren existieren Skripte zum Einladen von Paketen und Funktionen sowie ein Python-Skript zur Erstellung eines neuronalen Netzes.

Die Skripte enhalten aufgrund einer Sperrvermerks auf der Masterarbeit keine Daten. Sie sind somit nicht reproduzierbar. 

Im Folgenden wird eine kurze Übersicht über die in den Skripten enthaltenen Analysen gegeben:
  1. descriptive_analysis_ma.R:
     - Grafik zum Vergleich Nichtnutzende und Nutzende des Produkts
     - Korrelationsanalysen
     - Ausreißererkennung
     - LOESS-Grafiken
     - Produktregistrierungen im Zeitverlauf
     - Produktkorrelationsanalysen
  2. train_test_models_ma.R
     - Einladen und Aufbereitung der Datengrundlage
     - Training des Modells für unterschiedliche Algorithmen
     - Evaluation der Modellperformanz
     - Leistungsvergleich der Algorithmen
     - Merkmalsauswahl
     - Target Encoding Treshhold
     - Verteilung der Datentypen
