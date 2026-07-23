# in diesem Code werden die Ergebnisse des Skripts "multinoinale_logistische_Regression.py" manuell übertragen und visualisiert

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

answer_a = 'Wetter'
answer_b = 'Gipfel'
event_marker = 'Da'


ci_data_a = pd.DataFrame({
    'Variable': ['\'' + event_marker + '\' an 1. Position', '\'' + event_marker + '\' an 2. Position', 'männlich', '41-60 Jahre', 'über 60 Jahre'],
    'Estimate': [0.1293*100, -0.0957*100, -0.0391*100, 0.0093*100, 0.0472*100],
    'CI_lower': [0.009*100, -0.212*100, -0.137*100, -0.113*100, -0.131*100],
    'CI_upper': [0.250*100, 0.021*100, 0.058*100, 0.132*100, 0.225*100]
})
ci_data_a['Category'] = answer_a

ci_data_b = pd.DataFrame({
    'Variable': ['\'' + event_marker + '\' an 1. Position', '\'' + event_marker + '\' an 2. Position', 'männlich', '41-60 Jahre', 'über 60 Jahre'],
    'Estimate': [-0.2068*100, 0.0375*100, 0.0992*100, 0.0834*100, 0.0174*100],
    'CI_lower': [-0.339*100, -0.052*100, 0.019*100, -0.013*100, -0.130*100],
    'CI_upper': [-0.075*100, 0.127*100,  0.180*100,  0.180*100, 0.165*100]
})
ci_data_b['Category'] = answer_b

ci_data_c = pd.DataFrame({
    'Variable': ['\'' + event_marker + '\' an 1. Position', '\'' + event_marker + '\' an 2. Position', 'männlich', '41-60 Jahre', 'über 60 Jahre'],
    'Estimate': [0.0943*100, 0.0406*100, -0.0534*100, -0.0994*100, -0.1217*100],
    'CI_lower': [-0.014*100, -0.065*100, -0.143*100, -0.217*100, -0.301*100],
    'CI_upper': [0.203*100, 0.146*100, 0.036*100, 0.018*100, 0.058*100]
})
ci_data_c['Category'] = 'beide'

all_data = pd.concat([ci_data_a, ci_data_b, ci_data_c])

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

colors = {answer_a: '#00306D', answer_b: '#b51c1d', 'beide': '#727777'}
dodge_offset = 0.15

variables = ci_data_a['Variable'].unique()
y_ticks = np.arange(len(variables))

for i, var in enumerate(variables):
    a_row = all_data[(all_data['Variable'] == var) & (all_data['Category'] == answer_a)]
    b_row = all_data[(all_data['Variable'] == var) & (all_data['Category'] == answer_b)]
    c_row = all_data[(all_data['Variable'] == var) & (all_data['Category'] == 'beide')]
    
    ax.plot([a_row['CI_lower'], a_row['CI_upper']], [i - dodge_offset, i - dodge_offset], color=colors[answer_a], linewidth=3, solid_capstyle='round')
    ax.plot(a_row['Estimate'], i - dodge_offset, 'o', color='white', markeredgecolor=colors[answer_a], markersize=8)

    ax.plot([b_row['CI_lower'], b_row['CI_upper']], [i, i], color=colors[answer_b], linewidth=3, solid_capstyle='round')
    ax.plot(b_row['Estimate'], i, 'o', color='white', markeredgecolor=colors[answer_b], markersize=8)

    ax.plot([c_row['CI_lower'], c_row['CI_upper']], [i + dodge_offset, i + dodge_offset], color=colors['beide'], linewidth=3, solid_capstyle='round')
    ax.plot(c_row['Estimate'], i + dodge_offset, 'o', color='white', markeredgecolor=colors['beide'], markersize=8)

ax.axvline(0, color='grey', linestyle='--')

ax.set_yticks(y_ticks)
ax.set_yticklabels(variables)

legend_elements = [
    Line2D([0], [0], color=colors[answer_a], lw=4, label='Kategorie 1 \'' + answer_a + '\''),
    Line2D([0], [0], color=colors[answer_b], lw=4, label='Kategorie 2 \'' + answer_b + '\''),
    Line2D([0], [0], color=colors['beide'], lw=4, label='beide')
]
ax.legend(handles=legend_elements, loc='best', title='Antwortkategorie')

ax.set_title('Marginaleffekte für Antwortkategorien \'' + answer_a + '\', \'' + answer_b + '\' und \'beide\'')
ax.set_xlabel('Änderung der Wahrscheinlichkeit (in Prozentpunkten)')
ax.set_ylabel('Prädiktor')

ax.invert_yaxis()
plt.tight_layout()
plt.savefig('./output/Marginaleffekte_Berg_' + event_marker + '.png')
plt.show()
