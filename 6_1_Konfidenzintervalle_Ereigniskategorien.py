import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def count_phrase(group):
    words = group["word"].str.lower().reset_index(drop=True)
    pos = group["pos"].str.lower().reset_index(drop=True)
    n = len(words)
    matches = 0
    
    # für Trigramme verwenden
   # if n >= 3:
    #    matches += (
             #((words.shift(0) == "in") & 
             #(words.shift(-1) == "diesem") & 
             #(words.shift(-2) == "augenblick")) #|
             #((words.shift(0) == "in") & 
             #(words.shift(-1) == "diesem") & 
             #(words.shift(-2) == "moment")) #|
     #        ((words.shift(0) == "in") & 
      #       (words.shift(-1) == "dem") & 
       #      (words.shift(-2) == "augenblick")) #|
             #((words.shift(0) == "in") & 
             #(words.shift(-1) == "dem") & 
             #(words.shift(-2) == "moment"))
        #    ).sum()
    
    # für Bigramme verwenden
    #if n >= 2:
     #   matches += (
      #      ((words.shift(0) == "auf") & 
       #     (words.shift(-1) == "einmal"))
        #).sum()

   # if n >= 2:
    #    matches += (
     #       (words.shift(0) == "da") &
      #      (pos.shift(-1) == "verb")
       # ).sum()
    
    # für Unigramme verwenden
    if n >= 1:
        matches += (words == "plötzlich").sum()
    
    return pd.Series({
        "length": n,
        "frequency": matches,
        "frequency per million": (matches / n) * 1e6 if n > 0 else 0
    })

df = pd.read_csv("./texte/corpus_change_of_state.csv")
change_of_state_grouped = df.groupby(["text_id"])

df = pd.read_csv("./texte/corpus_process.csv")
process_grouped = df.groupby(["text_id"])

df = pd.read_csv("./texte/corpus_stative_event.csv")
stative_event_grouped = df.groupby(["text_id"])

df = pd.read_csv("./texte/corpus_non_event.csv")
non_event_grouped = df.groupby(["text_id"])

change_of_state_results = change_of_state_grouped.apply(count_phrase).reset_index()
process_results = process_grouped.apply(count_phrase).reset_index()
stative_event_results = stative_event_grouped.apply(count_phrase).reset_index()
non_event_results = non_event_grouped.apply(count_phrase).reset_index()

frequencies = [change_of_state_results["frequency per million"]] + [process_results["frequency per million"]] + [stative_event_results["frequency per million"]] + [non_event_results["frequency per million"]]

intervals = []
alpha = 0.05
n_bootstrap = 10000

for freq in frequencies:
    bootstrap_means = np.random.choice(freq, size=(n_bootstrap, len(freq)), replace=True).mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [alpha/2, 1 - alpha/2])
    mean = np.mean(freq)
    intervals.append([lower, mean, upper])

    data = np.array(intervals)
x = np.arange(1, len(data) + 1)
labels = ["Change of State", "Process", "Stative Event", "Non-Event"]

plt.figure(figsize=(10, 6))
plt.errorbar(
    x, data[:, 1], 
    yerr=[data[:, 1] - data[:, 0], data[:, 2] - data[:, 1]],
    fmt="-o", capsize=5, color="#00305D",
    label="Durchschnitt ± 95% Konfidenzintervall",
    linestyle="none"
)

plt.xticks(x, labels)
plt.xlabel("Kategorie")
plt.ylabel("Frequenz pro Million")
plt.title("Frequenz von 'plötzlich' mit 95% Konfidenzintervallen nach Ereigniskategorien") #je nach Ereignismarker anpassen
plt.legend()
plt.grid(True, alpha=0.7)

plt.savefig("./output/plötzlich_Event_Kategorien.png")#je nach Ereignismarker anpassen
plt.show()

print(intervals)