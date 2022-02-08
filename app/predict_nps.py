import numpy as np
import pandas as pd
import numpy as np
import regex as re
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from uploading import df

dict = {'Quel âge avez-vous ?': 'Age',
        'Quels sont vos usages de transport en commun': 'Usages',
        'De quel département êtes-vous ?': 'Dépratement',
        'À quelle fréquence utilisez-vous les transports routiers de votre région ?': 'Fréquence',
        'Quelle application utilisez-vous pour réaliser un itinéraire de transport ?':'Apps',
        'Saviez-vous que vous pouviez réaliser un itinéraire depuis le portail de la Nouvelle Aquitaine ? ': 'Itinéraire',
        ' Aller sur le site transport de la NA, puis indiquez quelle est votre niveau appréciation globale du portail ?': 'Appréciation-portail',
        'Que pensez-vous de l’esthétisme du site ?': 'Esthétisme',
        'Pourriez-vous simuler un itinéraire,  puis évaluer votre niveau de satisfaction du service': 'Itinéraire-satisfaction',
        'Quelle est la probabilité que vous recommandiez le portail de la Nouvelle Aquitaine à un ami ou un collègue ? ': 'NPS',
        '1':'Utilisation-fréquente',
        '2':'systeme-complexe',
        '3':'systeme-facile',
        '4':'support-specialiste',
        '5':'fonctions-intégrées',
        '6':'systeme_incoherent',
        '7':'facile-apprendre',
        '8':'systeme-contraignant',
        '9':'confiance',
        '10':'familiarisation-difficile',
        }

data = df.drop(['Horodateur','Comment décririez-vous votre usage des transports routiers en NA ?','En allant sur le portail de la nouvelle Aquitaine, quelles sont les informations que vous souhaiteriez trouver ?','Qu’est-ce que vous avez apprécié ?','Qu’est-ce que vous avez le plus aimé dans le design du site ?','Qu’est-ce que tu avez le moins aimé dans le design du site ?',"Comment pourrions-nous améliorer le portail d'un point de vue accessibilité ?",'sus items','Souhaitez-vous partager un commentaire, remarque ou suggestion ?'],axis=1)    
data.rename(columns=dict,
        inplace=True)

#drop rows that dont contain NPS and split data (features + Label)
data = data.drop(index=[0,1,2,3], axis=0)
data = data.reset_index()
data = data.drop(['index'],axis=1)

def processAge(ages):
    for i in range(len(ages)):
        ages[i] = int(re.sub(r'(?P<age>[0-9]+).*',r"\g<age>",ages[i]))
    return ages

#processing text features: Itinéraire and Fréquence

pipeline = Pipeline([
("imp", SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
("cat", OrdinalEncoder(categories=[['Non, je ne le savais pas', 'Oui, je le savais'],['Plusieurs fois par semaine', 'Quelques fois par an',
        'Quelques fois par mois','Jamais']]))
])

Itinieraire_frequence_pipeline = ColumnTransformer([
("cat", pipeline, ["Itinéraire","Fréquence"]),
])

data_prepared = Itinieraire_frequence_pipeline.fit_transform(data)
data[['Itinéraire','Fréquence']] = pd.DataFrame(data_prepared, columns=['Itinéraire','Fréquencé'], index=data.index)



#processing text features: Usages and Apps
mlb = MultiLabelBinarizer()
mlb2 = MultiLabelBinarizer()
pattern = re.compile('(?<!sport|courses), ')
classes = ['Activités de la vie quotidienne (aller au sport, faire les courses, ...)',
       'Déplacement professionnel occasionnel',
       'Déplacement professionnel quotidien', 'Tourisme','Transport scolaire']

def processUsagesApps(df):
       apps = df['Apps'].map(lambda x: x.split(', '))
       usages = df['Usages'].map(lambda x: [y if y in classes else 'Autre Usage' for y in pattern.split(x)])
       encoded1 = pd.DataFrame(mlb.fit_transform(apps), columns=mlb.classes_, index=df.index)
       encoded2 = pd.DataFrame(mlb2.fit_transform(usages), columns=mlb2.classes_, index=df.index)
       df[encoded1.columns] = encoded1
       df[encoded2.columns] = encoded2
       df.drop(['Usages','Apps'], axis=1, inplace=True)
       return df
       
def processNewUsagesApps(df):
       apps = df['Apps'].map(lambda x: x.split(', '))
       usages = df['Usages'].map(lambda x: [y if y in classes else 'Autre Usage' for y in pattern.split(x)])
       encoded1 = pd.DataFrame(mlb.transform(apps), columns=mlb.classes_, index=df.index)
       encoded2 = pd.DataFrame(mlb2.transform(usages), columns=mlb2.classes_, index=df.index)
       df[encoded1.columns] = encoded1
       df[encoded2.columns] = encoded2
       df.drop(['Usages','Apps'], axis=1,inplace=True)
       return df

y = data[['NPS']]
X = data.drop('NPS',axis=1)
       
#fill missing numerical values 
num_pipeline = Pipeline([
('imput', SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
('std_scaler', StandardScaler()),
])
X_array = num_pipeline.fit_transform(X)
X = pd.DataFrame(X_array, columns=X.columns, index=X.index, dtype=float)
final_data = pd.concat([X,y], axis=1)

def fullNewDataProcessing(df):
    df_prepared = Itinieraire_frequence_pipeline.transform(df)
    df[['Itinéraire','Fréquence']] = pd.DataFrame(df_prepared, columns=['Itinéraire','Fréquencé'], index=df.index)
    df = processNewUsagesApps(df)
    df_array = num_pipeline.transform(df)
    df = pd.DataFrame(df_array, columns=df.columns, index=df.index)
    return df

