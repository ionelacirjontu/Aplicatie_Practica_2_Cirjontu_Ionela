import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

date = pd.read_csv("dataset.csv")

date['cazareDisponibila'] = date['Accommodation_Available'].map({'Yes': 1, 'No': 0})

codificatorCategorie = LabelEncoder()
date['categorie'] = codificatorCategorie.fit_transform(date['Category'])

codificatorTara = LabelEncoder()
date['tara'] = codificatorTara.fit_transform(date['Country'])

date['VenitPeVizitator'] = date['Revenue'] / date['Visitors']

etichete = date['Rating'].values

caracteristici = date[['tara', 'categorie', 'Visitors', 'Revenue', 'cazareDisponibila', 'VenitPeVizitator']].values

xAntrenament, xTest, yAntrenament, yTest = train_test_split(caracteristici, etichete, test_size=0.2, random_state=10)

clasificatorSlab = DecisionTreeRegressor(max_depth=3)

adaBoost = AdaBoostRegressor(estimator=clasificatorSlab, n_estimators=50, random_state=10)

adaBoost.fit(xAntrenament, yAntrenament)

taraFixata = 5

dateTaraFixata = date[date['tara'] == taraFixata]

caracteristiciTara = dateTaraFixata[['tara', 'categorie', 'Visitors', 'Revenue', 'cazareDisponibila', 'VenitPeVizitator']].values

dateTaraFixata['ratingPrediccionat'] = adaBoost.predict(caracteristiciTara)

ranking = dateTaraFixata.groupby('categorie')['ratingPrediccionat'].mean().sort_values(ascending=False)

numeTara = codificatorTara.inverse_transform([taraFixata])[0]

print(f"\nIerarhia categoriilor pentru tara {numeTara}:")

for idx, (categorie, scor) in enumerate(ranking.items(), 1):
    print(f"{idx}. Categoria {codificatorCategorie.inverse_transform([categorie])[0]}: {scor:.4f}")
