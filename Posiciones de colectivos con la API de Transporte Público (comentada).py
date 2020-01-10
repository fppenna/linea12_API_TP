#!/usr/bin/env python
# coding: utf-8

# # API de Transporte Público Ciudad de Buenos Aires

# #### ¿Qué es una API (*application programming interface*)?
# Una API es un conjunto de funciones predefinidas que se utilizan para el desarrollo de aplicaciones, ofrecidas por una biblioteca para ser utilizadas por otro software. Uno de sus principales propósitos consiste en proporcionar funciones de uso general de forma que los desarrolladores hagan uso de las mismas para evitar escribir código desde cero.
# 
# #### La API de Transporte Público de Buenos Aires
# Tiene por objetivo proveer los datos abiertos del transporte público de la Ciudad en formato estandarizado y en **tiempo real**.
# 
# Algunos de los conjuntos de datos centrales multimodales incluidos y disponibles para los desarrolladores son:
# 
# > - Planificación del viaje (actual y futuro)
# > - Estado (actual y futuro)
# > - Interrupciones (actuales) y trabajos planificados (futuro)
# > - Predicciones de llegada / salida (instantánea y websockets)
# > - Horarios
# > - Puntos de embarque e instalaciones
# > - Rutas y líneas (topología y geográfica)
# 
# 

# ### Paso 1: Importar las Bibliotecas necesarias
# 

# In[1]:



import pandas as pd
import numpy as np

# This module provides regular expression matching operations
import re
# To make HTTP requests
import requests

# To transform json objects into DataFrame
from pandas.io.json import json_normalize

# To transform STR objects into json
import json

# To parse str objects into date and time objects
import datetime

# import urllib.request   ----> Otra libreria para consultar http nativa de python

# To get current time
import time

# To pause the loops
from time import sleep

# Make graphs
# !pip install plotly
import plotly.express as px


# In[2]:


url='https://apitransporte.buenosaires.gob.ar/colectivos/vehiclePositionsSimple?agency_id=6&client_id=a0119e3e5ebf4fb4a0cbdea04de12037&client_secret=19967Bb82e7e4C7bab2CA49e3d631C3E' 
pattern="({.+?})"
response= requests.get(url)


# In[3]:


response.text


# In[4]:


## construir Data Frame a partir de la consulta a la api.


# In[5]:


re.findall(pattern, response.text.replace('\n',''))[0]


# In[6]:


json1=re.findall(pattern, response.text.replace('\n',''))[5]

json1=json.loads(json1)
json_normalize(json1)


# In[7]:


type(json1)


# In[8]:


## como extraer datos de la api dejando correr el loop por tiempo indeterminado.
## por cantidad de consultas
## por límite de tiempo


# ### Construcción de la URL a consultar

# > **1-** Ingreso a *https://www.buenosaires.gob.ar/desarrollourbano/transporte/apitransporte/api-doc*
# 
# > **2-** Selecciono la opción */colectivos/vehiclePositions*: Devuelve la posición de los vehículos monitoreados actualizada cada 30 segundos 
# 
# > **3-** Ingreso las credenciales "client_id" y "client_secret" y el parámetro "agency_id" (para seleccionar la línea de colectivos que me interesa)
# 
# > **4-** Selecciono "GET" y copio la URL que se construye.

# Creo un objeto llamado url de tipo STR con la url que copié en la documentación de la API.
# 
# Notar que se construye concatenando la consulta que nos interesa **"vehiclePositionsSimple"** con el parámetro **"agency_id"** y las credenciales **"client_id"** y **"client_secret"**

# In[9]:


url='https://apitransporte.buenosaires.gob.ar/colectivos/vehiclePositionsSimple?agency_id=6&client_id=a0119e3e5ebf4fb4a0cbdea04de12037&client_secret=19967Bb82e7e4C7bab2CA49e3d631C3E' 


# ### Alternativa para construir la url
# 
# Creo seteo el parametro que me interesa y las credenciales de la API en objetos del tipo STR para luego completar las URL que se encuentra en otro objeto del tipo STR
# 

# In[10]:


# Option 2: Create a function
def url_bus(id_agency, user, psw):
    
    url= 'https://apitransporte.buenosaires.gob.ar/colectivos/vehiclePositionsSimple?agency_id='+str(id_agency)+'&client_id='+str(user)+'&client_secret='+str(psw) 
    
    return url    


# In[11]:


# Option 3: Set parameters & credentials in objects

agency_id= '6'

client_id= 'a0119e3e5ebf4fb4a0cbdea04de12037'

client_secret= '19967Bb82e7e4C7bab2CA49e3d631C3E'

url_incompleta= 'https://apitransporte.buenosaires.gob.ar/colectivos/vehiclePositionsSimple?agency_id={}&client_id={}&client_secret={}' 

# Con '.format' agrego otros strings al string principal 'url' 
# agregando como parametros lo que quiero que aparezca entre los corchetes

url_buses= url.format(agency_id,
                      client_id,
                      client_secret)


# ## Loop para recolectar los datos de la API en un período de tiempo

# In[9]:


# Create an empty Data Frame
agency_6=pd.DataFrame()

# this pattern captures all text in {}
pattern='({.*?})'

# 60 sec * 10 min
timeout = time.time() + 60*10
 
while True:
    
    # API requests
    response= requests.get(url)
    
    # Get text from API response
    text_api= response.text
    
    # Capture text in {} from the API response
    data_buses=re.findall(pattern, text_api.replace('\n',''))
   
    # For loop to append data from all bus units to empty Data Frame
    for i in range(len(data_buses)):
        
        # Convert to JSON (or dict) each row of the API response text 
        json_buses=json.loads(data_buses[i])
        
        # Convert to DataFrame and append to linea_12 DF 
        agency_6=agency_6.append(json_normalize(json_buses), ignore_index=True)
       
    print(agency_6)
    
    if time.time() > timeout:
        
        print('Fin del tiempo de ejecución')
        
        break
        
    ## Wait 30 seconds to get new data from the API               
    sleep(30)
    


# In[12]:


agency_6=pd.read_csv('agency_6.csv')


# ## Los datos recolectados de la API estan en agency_6

# In[13]:


agency_6.info()


# ### Análisis del Dataframe construido obteniendo datos de la API

# #### Metadatos

# >- **'agency_id'**: Número de referencia de la línea
# 
# >- **'agency_name'**: Nombre de la empresa que gestiona la línea
# 
# >- **'direction'** : Dirección a la que se dirige (Dummy)
# 
# >- **'id'**: Id del interno de la línea
# 
# >- **'latitude'**: Latitud de la ubicación del colectivo
# 
# >- **'longitude'**: Longitud de la ubicación del colectivo
# 
# >- **'route_id'**: Identificación de la línea perteneciente a la empresa
# 
# >- **'route_short_name'**: Nombre de la línea perteneciente a la empresa
# 
# >- **'speed'**: Velocidad
# 
# >- **'timestamp'**: Hora de la consulta (en S)
# 
# >- **'trip_headsign'**: Destino anunciado del viaje

# In[14]:


# Parseo la columna timestamp
agency_6['timestamp']=pd.Series([pd.Timestamp(t, unit='s') for t in agency_6['timestamp']])


# In[15]:


# Filtro los datos de la linea 12
linea_12=agency_6[agency_6['route_id']==63]


# In[16]:


linea_12=linea_12.reset_index(drop=True)
# cuento la cantidad de registros de cada uno de los internos
linea_12['id'].value_counts()


# #### Cantidad de internos circulando

# In[17]:


print('La cantidad de internos circulano en un período de 30 min es ', str(len(linea_12['id'].unique())))


# #### Cantidad de registros de cada interno

# In[18]:


internos_registro=pd.DataFrame(linea_12['id'].value_counts()).reset_index()


# In[19]:


internos_registro.columns=['interno','cantidad_registros']
internos_registro


# In[21]:


# Alternativa
# linea_12.groupby('id').count().loc[:,'agency_id']


# #### Reviso si en 30 minutos hay registros de internos que hacen viajes de ida y de vuelta

# Calculo el promedio de la columna "Direction" que contiene valores 1 y 0, agrupando por el ID del interno. El resultado asociado a cada uno de los internos será la proporcion de marcas GPS, registros o respuestas de la API en dirección "1". 
# 
# Si un interno tiene asociado un promedio "0", todas las respuestas de la API de ese interno tienen dirección "0"
# Si un interno tiene asociado un promedio "1", todas las respuestas de la API de ese interno tienen dirección "1"
# Si un interno tiene asociado un promedio entre 0 y 1, los registros de la posicion de ese interno corresponden a ambas direcciónes, "0" y "1".

# In[30]:


pd.DataFrame(linea_12.groupby('id').mean()['direction'].sort_values())


# #### Calculo la velocidad promedio de los internos cuando no estan detenidos y las velocidades máximas. 

# In[44]:


print('La velocidad promedio de los internos cuando no estan detenidos es ',str(round(linea_12['speed'][linea_12['speed']!=0].mean(), 2)))


# In[61]:


print('La velocidad máxima alcanzada por un interno es,',str(round(linea_12['speed'].max(),2)))


# In[8]:


## bonus encontrar clusters en los que se encuentren mas puntos para identificar paradas/semaforos. 


# ## Visualizacion de de la Línea 12

# In[ ]:





# In[62]:


linea_12


# #### Todos los puntos de posición conectado por líneas

# In[63]:


fig = px.line_mapbox(linea_12, lat="latitude", lon="longitude", color="id", zoom=3, height=300)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=11, mapbox_center_lat = -34.6,
    margin={"r":1,"t":0,"l":0,"b":0})

fig.show()


# #### Todos los puntos de posición de los colectivos pintados según la velocidad a la que transitaban
# 

# In[67]:



fig = px.scatter_mapbox(linea_12, lat="latitude", lon="longitude", color="speed", zoom=3, height=300)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=11, mapbox_center_lat = -34.6,
    margin={"r":1,"t":0,"l":0,"b":0})

fig.show()


# #### Selecciono las velocidades = 0 para detectar paradas

# In[70]:


linea_12_0=linea_12[linea_12['speed']==0]


# In[71]:



fig = px.scatter_mapbox(linea_12_0, lat="latitude", lon="longitude", color="speed", zoom=3, height=300)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=11, mapbox_center_lat = -34.6,
    margin={"r":1,"t":0,"l":0,"b":0})

fig.show()


# ## DB SCAN para detectar paradas

# #### Preprocesamiento de los datos geograficos para luego clusterizar

# In[72]:


# Primero centramos los datos
# distancia de cada observacion a la media
linea_12_0['lat_center'] = linea_12_0['latitude'] - np.mean(linea_12_0['latitude']) 
# distancia de cada observacion a la media
linea_12_0['lon_center'] = linea_12_0['longitude'] - np.mean(linea_12_0['longitude']) 


# In[74]:


def lat_a_metros(x):
    """Latitude:  1 grado = 110.54 km"""
    return x*110540

def lon_a_metros(x,cos_mean_lat):
    """Longitude: 1 grado = 111.320*cos(latitude) km"""
    return x*111320*cos_mean_lat


# In[75]:


cos_m_lat = np.cos(np.deg2rad(np.mean(linea_12_0['longitude'])))
print(cos_m_lat)


# In[77]:


linea_12_0['lat_metros'] = linea_12_0['lat_center'].apply(lambda x: round(lat_a_metros(x)))

linea_12_0['lon_metros'] = linea_12_0['lon_center'].apply(lambda x: round(lon_a_metros(x,cos_m_lat)))


# #### Entreno Modelo DBSCAN

# In[78]:


from sklearn.cluster import DBSCAN, KMeans


# In[86]:


dbscn = DBSCAN(eps = 10, min_samples = 7).fit(linea_12_0[['lat_metros','lon_metros']])


# In[87]:


dbscn


# In[88]:


labels = dbscn.labels_

len(pd.Series(labels).unique())


# In[89]:


## El  -1 no es un cluster. Lo elimino cuando cuento los cluster. El -1 tiene asignado los puntos q no pertenecen a un cluster
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


# In[90]:


n_clusters_


# In[91]:


linea_12_0['labels'] = labels

clusters = linea_12_0.loc[linea_12_0['labels']!=-1].copy()

clusters.sort_values('labels')


# #### Grafico los clusters y los puntos

# In[92]:



fig = px.scatter_mapbox(clusters, lat="latitude", lon="longitude", color="labels", zoom=3, height=300)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=11, mapbox_center_lat = -34.6,
    margin={"r":1,"t":0,"l":0,"b":0})

fig.show()

