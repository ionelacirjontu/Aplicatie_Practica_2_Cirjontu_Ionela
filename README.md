# Aplicatie_Practica_1_Cirjontu_Ionela SOLUTIA 3

Acesta este un algoritm care foloseste AdaBoost cu arbori de decizie pentru ranking.


### 1. Importarea bibliotecilor

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
```
NumPy este folosit pentru manipularea si calculul numeric al datelor.
Pandas este folosit pentru manipularea datelor intr-un format tabular, similar cu tabelele de baze de date.
Sklearn este folosit pentru antrenarea modelului

### 2. Citirea datelor din csv
```
date = pd.read_csv("dataset.csv")

```

### 3.Preluarea coloanei Accomodationa_Available si codificare yes=1, no =0
```
date['cazareDisponibila'] =date['Accommodation_Available'].map({'Yes': 1, 'No': 0})
```

### 4.Codificarea coloanelor cu text Category si Country in numere
```
codificatorCategorie =LabelEncoder()
date['categorie'] =codificatorCategorie.fit_transform(date['Category'])

codificatorTara = LabelEncoder()
date['tara'] = codificatorTara.fit_transform(date['Country'])
```

### 5.Introducerea atributului nou venitPeVizitator
```
date['VenitPeVizitator'] = date['Revenue'] / date['Visitors']
```

### 6.Eticheta este ratingul
```
etichete = date['Rating'].values
```
### 7. Atributele folosite de model
```
caracteristici = date[['tara','categorie','Visitors','Revenue','cazareDisponibila','VenitPeVizitator']].values
```

### 8.Impartirea setului in 80% set de antrenament si 20% set de testare
```
xAntrenament,xTest,yAntrenament,yTest = train_test_split(caracteristici,etichete,test_size=0.2,random_state=10)
```

### 9.Clasificatorul slab folosit este un arbore de adancime de maxim 3
```
clasificatorSlab=DecisionTreeRegressor(max_depth=3)
```

### 10.Antrenarea modelului AdaBoost
```
adaBoost=AdaBoostRegressor(estimator=clasificatorSlab,n_estimators=30,random_state=10)
adaBoost.fit(xAntrenament, yAntrenament)
```

### 11.Realizarea rankingul pentru tara cu id-ul 5
```
taraFixata = 5

dateTaraFixata = date[date['tara'] == taraFixata]

caracteristiciTara = dateTaraFixata[['tara', 'categorie', 'Visitors', 'Revenue', 'cazareDisponibila', 'VenitPeVizitator']].values

dateTaraFixata['ratingPrediccionat'] = adaBoost.predict(caracteristiciTara)

ranking = dateTaraFixata.groupby('categorie')['ratingPrediccionat'].mean().sort_values(ascending=False)

numeTara = codificatorTara.inverse_transform([taraFixata])[0]

print(f"\nIerarhia categoriilor pentru tara {numeTara}:")

for idx, (categorie, scor) in enumerate(ranking.items(), 1):
    print(f"{idx}. Categoria {codificatorCategorie.inverse_transform([categorie])[0]}: {scor:.4f}")
```


# Aplicatie_Practica_2_Cirjontu_Ionela
