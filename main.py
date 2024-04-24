import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ocitavamo podatke iz csv fajla (kaggle)
engine_data = pd.read_csv('engine_data.csv')

# Ukupan broj slucajeva u csv fajlu
num_of_engine_conditions = len(engine_data['Engine Condition'])

# Izvlacimo podatke iz kolone "Engine Condition" -> (0 - kvarni, 1 - ispravni) bool
values = engine_data['Engine Condition'].value_counts()

# Procenat pokvarenih motora 
failed_engines = values[0] / num_of_engine_conditions * 100

# Procenat ispravnih motora
working_engines = values[1] / num_of_engine_conditions * 100

# Prikazujemo njihove vrednosti u procentima
print(f'Procenat pokvarenih motora: {round(failed_engines, 2)}%')
print(f'Procenat radnih motora: {round(working_engines, 2)}%')

# Odvajamo karakteristike
engine_features = engine_data.drop('Engine Condition', axis=1)
engine_labels = engine_data['Engine Condition']

# Vizuelno predstavljamo distribuciju svih atributa uz pomoc pairplot-a
# sns.pairplot(engine_data, hue='Engine Condition')
# plt.show()

# Racunamo korelacije koeficijenata
corr_matrix = engine_data.corr()
corr_values = corr_matrix['Engine Condition'].sort_values(ascending=False)

# Odavde vidimo da stanje motora (Engine Condition) ima najvecu pozitivnu korelaciju sa pritiskom goriva (Fuel Pressure)
# Dok za njegove obrtaje (Engine RPM) ima najvecu negativnu korelaciju
# print(corr_values)

# Sada spajamo najvecu pozitivnu korelaciju (Fuel Pressure) sa najvecom negativnom korelacijom (Engine RPM)
sns.jointplot(engine_data, x='Fuel pressure', y='Engine rpm', hue='Engine Condition')
plt.show()





