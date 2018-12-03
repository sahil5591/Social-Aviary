#######    extracting tweets


import tweepy #https://github.com/tweepy/tweepy
import csv
import pandas as pd
# Used for progress bar
import time
import sys

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print (__version__)
    

#Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""

OAUTH_KEYS = {'consumer_key':consumer_key, 'consumer_secret':consumer_secret,
 'access_token_key':access_key, 'access_token_secret':access_secret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

search = tweepy.Cursor(api.search, q='manmohan').items(15000)
print (search)
# Create lists for each field desired from the tweets.
sn = []
text = []
timestamp =[]
for tweet in search:
    print (tweet.user.screen_name)
    print (tweet)
    timestamp.append(tweet.created_at)
    sn.append(tweet.user.screen_name)
    text.append(tweet.text)
    
# Convert lists to dataframe
df = pd.DataFrame()
df['timestamp'] = timestamp
df['sn'] = sn
df['text'] = text    

userids = list(df['sn'].unique())

###userids = pd.read_csv("userid.csv",names = ['userid'])


# Initialize dataframe of users that will hold the edge relationships
dfUsers = pd.DataFrame()
dfUsers['userFromName'] =[]
dfUsers['userFromId'] =[]
dfUsers['userToId'] = []
count = 0 
userids = pd.DataFrame({'userid':userids})
nameCount = len(userids.userid)

for name in userids.userid:
    # Build list of friends    
    currentFriends = []
    for page in tweepy.Cursor(api.friends_ids, screen_name=name).pages():
        
        currentFriends.extend(page)
    currentId = api.get_user(screen_name=name).id
    currentId = [currentId] * len(currentFriends)
    currentName = [name] * len(currentFriends)   
    dfTemp = pd.DataFrame()
    dfTemp['userFromName'] = currentName
    dfTemp['userFromId'] = currentId
    dfTemp['userToId'] = currentFriends
    dfUsers = pd.concat([dfUsers,dfTemp])
     # avoids hitting Twitter rate limit
    # Progress bar to track approximate progress
    time.sleep(120)
    count +=1
    per = round(count*100.0/nameCount,1)
    sys.stdout.write("\rTwitter call %s%% complete." % per)
    sys.stdout.flush()    
    a = dfUsers
fromId = dfUsers['userFromId'].unique()
dfChat = dfUsers[dfUsers['userToId'].apply(lambda x: x in fromId)]

# No more Twitter API lookups are necessary. Create a lookup table that we will use to get the verify the userToName
dfLookup = dfChat[['userFromName','userFromId']]
dfLookup1 = dfLookup.drop_duplicates()
dfLookup.columns = ['userToName','userToId']
dfCommunity = dfUsers.merge(dfLookup, on='userToId')

dfCommunity.to_csv('dfCommunity.csv',index = False,encoding='utf-8')

import plotly.plotly as py
from plotly.graph_objs import *

import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline 
    
    
df = pd.read_csv('dfCommunity.csv')
#df = df[index]
# Convert user ID from float to integer.
df.userFromId=df.userFromId.apply(lambda x: int(x))
df.userToId=df.userToId.apply(lambda x: int(x))

#G = nx.DiGraph()
G = nx.Graph()
G.add_nodes_from(df['userFromId'])
#G.add_edges_from(zip(df['userFromId'],df['userToId']))
temp = zip(df['userFromId'],df['userToId'])
G.add_edges_from(temp)

# Give nodes their Usernames
dfLookup = df[['userFromName','userFromId']].drop_duplicates()

dfLookup.head()
for userId in dfLookup['userFromId']:
    temp = dfLookup['userFromName'][df['userFromId']==userId]
    G.node[userId]['userName'] = temp.values[0]
nx.draw(G, pos=nx.spring_layout(G,k=.12),node_color='c',edge_color='k')



### only for di graph
pos=nx.spring_layout(G,k=.12)
centralScore = nx.betweenness_centrality(G)
inScore = G.in_degree()
outScore = G.out_degree()


#Define scatter_nodes() and scatter_edges()
#These functions create Plotly "traces" of the nodes and edges using the layout defined in "pos". Here, I have chosen to color the nodes by the betweenness centrality, but one might choose to vary size of the nodes instead, or vary by another characteristic such as degree
# Get a list of all nodeID in ascending order
nodeID = G.node.keys()
nodeID.sort()

def scatter_nodes(pos, labels=None, color='rgb(152, 0, 0)', size=8, opacity=1):
    # pos is the dict of node positions
    # labels is a list  of labels of len(pos), to be displayed when hovering the mouse over the nodes
    # color is the color for nodes. When it is set as None the Plotly default color is used
    # size is the size of the dots representing the nodes
    # opacity is a value between [0,1] defining the node color opacity

    trace = Scatter(x=[], 
                    y=[],  
                    mode='markers', 
                    marker=Marker(
        showscale=True,
        # colorscale options
        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
        colorscale='Greens',
        reversescale=True,
        color=[], 
        size=10,         
        colorbar=dict(
            thickness=15,
            title='Betweeenness Centrality',
            xanchor='left',
            titleside='right'
        ),
    line=dict(width=2)))
    for nd in nodeID:
        trace['x'].append(pos[nd][0])
        trace['y'].append(pos[nd][1])
        trace['marker']['color'].append(centralScore[nd])
    attrib=dict(name='', text=labels , hoverinfo='text', opacity=opacity) # a dict of Plotly node attributes
    trace=dict(trace, **attrib)# concatenate the dict trace and attrib
    trace['marker']['size']=size

    return trace    

def scatter_edges(G, pos, line_color='#a3a3c2', line_width=1, opacity=.2):
    trace = Scatter(x=[], 
                    y=[], 
                    mode='lines',
                   )
    for edge in G.edges():
        trace['x'] += [pos[edge[0]][0],pos[edge[1]][0], None]
        trace['y'] += [pos[edge[0]][1],pos[edge[1]][1], None]  
        trace['hoverinfo']='none'
        trace['line']['width']=line_width
        if line_color is not None: # when it is None a default Plotly color is used
            trace['line']['color']=line_color
    return trace  
             
# Node label information available on hover. Note that some html tags such as line break <br> are recognized within a string.
labels = []

for nd in nodeID:
      labels.append(G.node[nd]['userName'] + "<br>" + "Followers: " + str(inScore[nd]) + "<br>" + "Following: " + str(outScore[nd]) + "<br>" + "Centrality: " + str("%0.3f" % centralScore[nd]))
      
 
trace1=scatter_edges(G, pos)
trace2=scatter_nodes(pos, labels=labels)    

width=600
height=600
axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )
layout=Layout(title= '#HIC Community on Twitter',
    font= Font(),
    showlegend=False,
    autosize=False,
    width=width,
    height=height,
    xaxis=dict(
        title='HIC community',
        titlefont=dict(
        size=14,
        color='#7f7f7f'),
        showline=False,
        showticklabels=False,
        zeroline=False
    ),
    yaxis=YAxis(axis),
    margin=Margin(
        l=40,
        r=40,
        b=85,
        t=100,
        pad=0,
       
    ),
    hovermode='closest',
    plot_bgcolor='#EFECEA', #set background color            
    )


data=Data([trace1, trace2])

fig = Figure(data=data, layout=layout)

def make_annotations(pos, text, font_size=14, font_color='rgb(25,25,25)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = Annotations()
    for nd in nodeID:
        annotations.append(
            Annotation(
                text="",
                x=pos[nd][0], y=pos[nd][1],
                xref='x1', yref='y1',
                font=dict(color= font_color, size=font_size),
                showarrow=False)
        )
    return annotations  

fig['layout'].update(annotations=make_annotations(pos, labels))  
py.sign_in('', '')
py.iplot(fig, filename='#HIC') 


########### time series graphs

import pandas as pd
import numpy as np
import datetime as dt

a = pd.read_csv("tweets1.csv")
a['datetime'] = pd.to_datetime(a['datetime'])
a['date'] = a['datetime'].dt.date


b = a.drop_duplicates(['userid']) 

import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
b['datetime'] = pd.to_datetime(b['datetime'])
b['date'] = b['datetime'].dt.date


###  vetex growth graph
vetex_growth = b.groupby('date').count()

trace_high = go.Scatter(
                x=vetex_growth.index,
                y=vetex_growth['userid'],
                name = "vertex growth",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

data = [trace_high]

layout = dict(
    title = "Vertex GROWTH Over Time",
    xaxis = dict(
        range = ['2017-08-08','2017-08-16'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Vertex Growth Over Time")

## graph density plot
graph_density= a.groupby('date').count()
k = 2 * graph_density.userid 
j = vetex_growth.userid * vetex_growth.userid - 1
graph_density['desnity'] = k/j


trace_high = go.Scatter(
                x= graph_density.index,
                y= graph_density.desnity,
                name = "graph density",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

data = [trace_high]

layout = dict(
    title = "Graph Density Over Time",
    xaxis = dict(
        range = ['2017-08-08','2017-08-16'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Graph Density Over Time")

## avg degree
degree = list(G.degree_iter())
degree = pd.DataFrame(degree)
degree.columns = ['userid', 'degree']


avg_degree = dfCommunity.merge(degree, on='userFromId')
avg_degree =avg_degree.rename(columns={'userFromName': 'username'})

avg_degree1 = a.merge(avg_degree, on='username')
avg_degree1 = avg_degree1.drop_duplicates(['username'])
avg_degree2 = avg_degree1.groupby('date').count()
avg_degree3 = avg_degree1.groupby('date').sum()
k = avg_degree3.degree
j = avg_degree2.degree
avg_degree3['avgd'] = k/j

trace_high = go.Scatter(
                x= avg_degree3.index,
                y= avg_degree3.avgd,
                name = "Average Degree",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

data = [trace_high]

layout = dict(
    title = "Average Degree Over Time",
    xaxis = dict(
        range = ['2017-08-08','2017-08-16'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Average Degree Over Time")

## median degree graph
med_degree = avg_degree1.groupby('date').median()

trace_high = go.Scatter(
                x= med_degree.index,
                y= med_degree.degree,
                name = "Median Degree",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

data = [trace_high]

layout = dict(
    title = "Median Degree Over Time",
    xaxis = dict(
        range = ['2017-08-08','2017-08-16'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Median Degree Over Time")

### graph
r = nx.Graph()
#G = nx.Graph()
r.add_nodes_from(df['userFromId'])
#G.add_edges_from(zip(df['userFromId'],df['userToId']))
temp = zip(df['userFromId'],df['userToId'])
r.add_edges_from(temp)

# Give nodes their Usernames
dfLookup = df[['userFromName','userFromId']].drop_duplicates()

dfLookup.head()
for userId in dfLookup['userFromId']:
    temp = dfLookup['userFromName'][df['userFromId']==userId]
    r.node[userId]['userName'] = temp.values[0]
nx.draw(r, pos=nx.spring_layout(r,k=.12),node_color='c',edge_color='k')



import pandas as pd
## clustering coefficient
clus = nx.clustering(r)
clus = pd.DataFrame(list(clus.items())) 
clus.columns = ['userFromId','clus_coeff']

clus_coeff = avg_degree1.merge(clus, on='userFromId')

clus_coeff1 = clus_coeff.groupby('date').mean()

trace_high = go.Scatter(
                x= clus_coeff1.index,
                y= clus_coeff1.clus_coeff,
                name = "Clustering Coefficient",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

data = [trace_high]

layout = dict(
    title = "Clustering Coefficient Over Time",
    xaxis = dict(
        range = ['2017-08-08','2017-08-16'])
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Clustering Coefficient Over Time")