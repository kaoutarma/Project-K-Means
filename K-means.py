from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
import pandas as pd
import configparser
import folium
#instatiation de Spark:
spark = SparkSession.builder \
    .master('local') \
    .appName('Bristol-City-bike') \
    .getOrCreate()
#utilisation du fichier de configuration pour récupérer les paths
config = configparser.ConfigParser()
config.read('properties.conf')
#Création des variables:
path_to_input_data = config['Bristol-City-bike']['Input-data']
path_to_output_data = config['Bristol-City-bike']['Output-data']
num_partition_kmeans = int(config['Bristol-City-bike']['Kmeans-level'])
#Importation du json:
bristol = spark\
            .read\
            .json(path_to_input_data)
bristol.show(5)
#création d'un nouveau data frame contenant seulement les variables latitude et longitude:
kmeans_df = bristol.select("latitude","longitude")
kmeans_df.show(5)
#k-means:
features = ('longitude','latitude')
kmeans = KMeans()\
            .setK(num_partition_kmeans)\
            .setSeed(1)
assembler = VectorAssembler(inputCols = features, outputCol = "features")
dataset = assembler\
            .transform(kmeans_df)
model = kmeans\
            .fit(dataset)
fitted = model\
            .transform(dataset)
#Les noms des colonnes de fitted 
fitted.columns #il s’agit de longitude, latitude, features, predictions.
#Détermination des longitudes et latitudes moyennes pour chaque groupe en utilisant spark DSL
meanByGroup = fitted\
                .groupBy(fitted.prediction)\
                .agg(mean('latitude').alias('LatMoyenne'),
                     mean('longitude').alias('LongMoyenne')
                     )\
                .orderBy('prediction')
meanByGroup.show()
#Détermination des longitudes et latitudes moyennes pour chaque groupe en utilisant spark SQL
fitted.createOrReplaceTempView("fittedSQL")
spark.sql("""
    select prediction,
           mean(latitude) as LatMoyenne,
           mean(longitude) as LongMoyenne
    from fittedSQL
    group by prediction
    order by prediction
""").show()
#Comparaison: on trouve les meme résultats avec les 2 méthodes
#Visualisation dans une carte avec le package folium
#Création de la dataframe nécessaire et initiation d'une carte vide:
def get_column(data,name):
    return data.select(name).rdd.flatMap(lambda x: x).collect()
groups = get_column(fitted, 'prediction')
latitudes = get_column(bristol,'latitude')
longitudes = get_column(bristol, 'longitude')
addresses = get_column(bristol, 'address')

df = pd.DataFrame({ 'add' : addresses,
                    'lat' : latitudes,
                    'lon' : longitudes,
                    'grp' : groups})
g = [df['lat'].mean(), df['lon'].mean()]
maps = folium\
        .Map(location=g,
               zoom_start=13)
#Ajout des marqueurs:
def add_city(row):
    if row['grp']==0:
        col='green'
    elif row['grp']==1:
        col='red'
    else:
        col='blue'
    folium.Marker(
        location= [row['lat'],row['lon']],
        popup=folium.Popup(row['add']),
        icon=folium.Icon(color=col),
    ).add_to(maps)
for index, row in df.iterrows():
    add_city(row)
#Ajout des centres de chaque groupe:
gData = meanByGroup.toPandas()
for index, row in gData.iterrows():
    folium\
        .Marker(
            location= [row['LatMoyenne'],row['LongMoyenne']],
            popup=folium.Popup(f"Center of group {int(row['prediction'])}"),
            icon=folium.Icon(color='beige'),
        ).add_to(maps)
#affichage de la carte:
maps
maps.save("output/Bristol-City-bike.html") #la carte sauvegardée sous format html
#Exportation de la data frame fitted après élimination de la colonne features, dans le répertoire associé:
fitted\
    .drop('features')\
    .write\
    .format("csv")\
    .mode("overwrite")\
    .save(path_to_output_data, header = 'true')
#arreter la session spark:
spark.stop()

