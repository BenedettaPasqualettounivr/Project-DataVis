#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px


# In[3]:


spotify_df = pd.read_csv('https://raw.githubusercontent.com/BenedettaPasqualettounivr/Project-DataVis/main/songs_normalize.csv?token=GHSAT0AAAAAABV3I3N5R4WXLJRCVM4CDGYAYV3JBFA')


# In[4]:


spotify_df.head()


# In[5]:


spotify_df.tail()


# In[6]:


spotify_df.shape


# In[7]:


spotify_df.columns


# In[8]:


spotify_df.describe().T


# In[12]:


#Now we can check the null values in dataset
spotify_df.isnull().sum().T


# In[9]:


#Checking the year of release present in the dataset:
spotify_df.year.unique()


# In[10]:


#We can see that there are some songs that are published outside the interval 2000-2019. So we can check the number of songs from these years present in the dataset
s1=(len(spotify_df.query('year==1998')))
s2=(len(spotify_df.query('year==1999')))
s3=(len(spotify_df.query('year==2020')))
s= s1+s2+s3
print("The total number of songs:",s)


# In[11]:


#Thus there are 42 songs from all 3 years combined. The goal of the dataset is to analyse data from the timeperiod [2000-2019] and hence any other data outside this timeline can be neglected
#Hence we would be removing all the data pertaining to these years from the dataframe
spotify_df_years_drop = spotify_df[(spotify_df['year'] <2000) | (spotify_df['year'] > 2019)].index
df = spotify_df.drop(spotify_df_years_drop)


# In[13]:


artist=df['artist'].value_counts()
artist


# In[14]:


tp_artists_songs= artist[:5]
tp_artists_name =artist[:5].index
fig = plt.figure(figsize = (10, 5))
plt.bar(tp_artists_name,tp_artists_songs,width = 0.4,color="forestgreen")
plt.xlabel("Artists")
plt.ylabel("No of Songs")
plt.title('Top Artists with Hit Songs',color = 'black',fontsize = 20)
plt.show()


# In[15]:


#The most popular artists:


# In[16]:


pop_artist = df.groupby('artist')[['artist','popularity']].sum().sort_values('popularity',ascending=False).head(10)
pop_artist


# In[17]:


plt.figure(figsize=(14,8))
plt.title("Most Popular Artists",fontsize=15)
c1 = sns.barplot(x="popularity",y=pop_artist.index,data=pop_artist,palette="Greens_r")
c1.bar_label(c1.containers[0],size = 15)
plt.show()


# In[18]:


#Rihanna seems to be the most popular artist with 25 songs of her to be listed as the top-hit in that particular year.
#Immediately followed by Drake who has 23 songs which has been declared as the top-hit over the past 19yrs (2000-2019)


# In[19]:


genre = df['genre'].value_counts()
genre


# In[20]:


plt.figure(figsize=(15,10))
sns.countplot(y='genre', data=df,palette="rocket")
plt.show()


# In[21]:


tp_genres=genre[:5]
tp_genres_names=genre[:5].index
fig = plt.figure(figsize = (10, 5))
plt.bar(tp_genres_names,tp_genres,width = 0.4,color='b')
plt.xlabel("Genres")
plt.ylabel("No of songs")
plt.title('Top 5 Genres or Combination of Genres which are hits',color = 'black',fontsize = 20)
plt.show()


# In[22]:


#Pop seems to be the most popular type of genre
#428 songs of the top-hits spotify songs since 2000-2019 belong to pop
#This is followed by hip-hop,pop which is the 2nd most popular conbination of genre


# In[23]:


genre_distribution = pd.DataFrame(spotify_df.genre.value_counts().rename_axis('Genre').reset_index(
    name='total'))
genre_distribution.head(10)


# In[24]:


plt.figure(figsize=(8,2))
fig = px.pie(genre_distribution.head(10), names = 'Genre',
      values = 'total', template = 'seaborn', title = 'Top 10 genres distribution')
fig.update_traces(rotation=90, pull = [0.2,0.06,0.06,0.06,0.06], textinfo = "percent+label")
fig.show()


# In[25]:


#Average Duration of Top Hit Songs for each year:


# In[26]:


#converting the duration of songs from milliseconds to minutes for easier understanding
def time_convert(ms):
    sec=ms/1000
    return f"{int(sec//60)}:{int(sec%60)}"
durations = df[['duration_ms','year']].groupby('year').mean().reset_index().iloc[0:20]
durations['min:sec'] = durations['duration_ms'].apply(time_convert)
durations


# In[47]:


# Creating a function that converts milliseconds to minutes and seconds
def ms_to_min_sec(ms):
    sec = ms/1000
    return f"{int(sec//60)}:{int(sec%60)}"

df['min:sec'] = df['duration_ms'].apply(ms_to_min_sec)


# In[48]:


songs_by_duration = df.sort_values("min:sec", ascending=False)[["song", "artist", "min:sec"]]
#Longest songs
songs_by_duration[:5]


# In[49]:


#Shortest songs
songs_by_duration[-5:]


# In[27]:


x=df["year"].unique()
x=sorted(x)
y=(durations["min:sec"])
x=pd.Series(x)
plt.figure(figsize=(12,8))
sns.lineplot(x,y,data=df)
plt.title('Change in the Duration of Song',fontsize=15)
plt.xlabel('Years',size=10)
plt.ylabel('Time',size=10)
plt.show()


# In[28]:


#Here we can find the general pattern that as the years pass by the average song duration has been decreasing


# In[33]:


#Explicity of songs
explicit_songs_or_not = (spotify_df.explicit.value_counts().rename_axis('Explicit').reset_index(name = 'songs'))
explicit_songs_or_not


# In[29]:


#Amount of Explicit Content in top-hits:


# In[30]:


data=df['explicit'].value_counts()
plt.figure(figsize=(10,8))
plt.title("Total Amount of Explicit Songs",fontsize=14)
plt.ylabel("No of Songs")
plt.xlabel("Explicitness")
data.plot(kind='bar',color=['lightblue', 'violet'])

plt.show()


# In[34]:


fig = px.pie(explicit_songs_or_not, names = ['Not explicit','Explicit'], 
             values = 'songs', template='seaborn',
            title = 'Percentage of explicit content in the top hits from 2000 to 2019')
fig.show()


# In[31]:


#Number of Top-hits which are explict over the years:


# In[32]:


song_yr_explicit = df.groupby(['year','explicit']).size().unstack(fill_value=0).reset_index()
song_yr_explicit.rename(columns={False:'Clean', True: 'Explicit'}, inplace=True)
plt.figure(figsize=(14,8))
plt.title("Top hits each year which are explicit",fontsize=15)
c1 = sns.barplot(x="year",y="Explicit",data=song_yr_explicit,palette="mako")
c1.bar_label(c1.containers[0],size = 15)
plt.show()


# In[35]:


#Number of songs released each year which became top-hits:


# In[36]:


songs_yr = df.year.value_counts().rename_axis('year').reset_index(name='songs') 
plt.figure(figsize=(14,8))
plt.title("No of top-hits released each year",fontsize=17)
c1 = sns.barplot(x="year",y="songs",data=songs_yr,palette="flare")
c1.bar_label(c1.containers[0],size = 15)
plt.show()


# In[37]:


#Analying the the top-hits from 1999 to 2019 we can see , 2012 had the maximum number of songs to make it to the top-hits, while the lowest was in 2000


# In[45]:


#Number of songs per year
df.groupby("year")["song"].count().plot()


# In[38]:


#Danceability:


# In[40]:


plt.figure(figsize=(10,10))
plt.title("Danceability of top-hits over the years:",size=15)
sns.distplot(df['danceability'],color="r")
plt.show()
print("Average danceability of top-hits:\n",df['danceability'].mean())
print("Maximum danceability of top-hits:\n")
df.loc[df['danceability']==df['danceability'].max()]


# In[41]:


#Correlation between various features:


# In[42]:


#creating a new dataframe to find the relation among numerical variables present in the dataset
df_new = df[['duration_ms', 'year', 'popularity', 'danceability', 'energy', 
     'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]


# In[43]:


plt.figure(figsize=(15,8))
sns.heatmap(df_new.corr(), annot = True, linewidths = .5, fmt = '.2f',cmap = "YlGn")
plt.title('Correlation among measures', size = 25)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12, rotation = 50)
plt.show()


# In[50]:


#Focus on Rihanna


# In[51]:


rihanna_df = df[df['artist'] == 'Rihanna']
rihanna_df.head()


# In[53]:


#How many songs she has
len(rihanna_df.song.unique())


# In[54]:


#Get the popularity and number of song per year (each block represent one music)
fig = px.bar(rihanna_df, x='year', y='popularity',title = 'Total Popularity of song by year')
fig.show()


# In[56]:


#Sum of popularity
px.histogram(rihanna_df, x="year", y="popularity",
                   hover_data=rihanna_df.columns)


# In[57]:


#Her first five Best Music
rihanna_df = rihanna_df.sort_values('popularity',ascending=False)
rihanna_df[['song','year','popularity','genre']].head()


# In[58]:


#See if there is a coorelation fields
plt.figure(figsize=(15,10))
sns.heatmap(df.corr())


# In[ ]:




