#!/usr/bin/env python
# coding: utf-8

# # DISSERTATION REPORT DATASET EVALUATION ON “#EndViolence UNICEF”

# # 1. ANALYSIS OF CRIME AGAINST WOMEN FROM 2001 TO 2014 IN INDIA

# # DATASET

This data is collated from https://data.gov.in. It has state-wise and district level data on the various crimes committed against women between 2001 to 2014.

Crimes that are include are :

'Rape'
'Kidnapping and Abduction'
'Dowry Deaths'
'Assault on women with intent to outrage her modesty'
'Insult to modesty of Women','Cruelty by Husband or his Relatives'
'Importation of Girls'
States and Union Territories
District
Year
# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


BHAVANA=pd.read_csv("crimes_against_women_2001-2014.csv")


# In[3]:


BHAVANA


# In[4]:


#REMOVING DUPLICATES
BHAVANA["STATE/UT"]=BHAVANA.apply(lambda row:row['STATE/UT'].replace(" ","").lower(),axis=1)
BHAVANA['STATE/UT'].replace("delhiut",'delhi',inplace=True) 


# In[5]:


BHAVANA.isnull().sum()


# In[6]:


#Dropping "Unnamed" Column
BHAVANA=BHAVANA.drop(['Unnamed: 0'],axis=1)


# In[7]:


BHAVANA.head()


# In[8]:


BHAVANA['total_crimes']=(BHAVANA['Rape']+BHAVANA['Kidnapping and Abduction']+BHAVANA['Dowry Deaths']+
                       BHAVANA['Assault on women with intent to outrage her modesty']+
                       BHAVANA['Insult to modesty of Women']+BHAVANA['Cruelty by Husband or his Relatives']+
                       BHAVANA['Importation of Girls'])


# In[9]:


value_count=list(BHAVANA["STATE/UT"].value_counts())
print("***** Value counts of STATES/UT *****")
print()
print(value_count)


# In[10]:


#total_crimes
crimes=['Rape','Kidnapping and Abduction','Dowry Deaths',
        'Assault on women with intent to outrage her modesty',
        'Insult to modesty of Women','Cruelty by Husband or his Relatives',
        'Importation of Girls','total_crimes']

BHAVANA1=pd.DataFrame()
for i in crimes:
    BHAVANA_crimes=BHAVANA.groupby(['Year'])[i].sum()
    BHAVANA1[i]=BHAVANA_crimes

print("***** Total number of crimes from 2001 to 2014 *****")
print()
BHAVANA1


# In[11]:


def plotting_cat_features(nrows,ncols,cat_columns):
    
    f,ax=plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,25))
    f.patch.set_facecolor('#F2EDD7FF')
    #Setting background and foreground color
    for i in range(0,nrows):
        for j in range(0,ncols):
            ax[i][j].set_facecolor('#F2EDD7FF')

    #Plotting count plot 
    for i in range(0,nrows):
        for j in range(0,ncols):
            a1=sns.barplot(data=BHAVANA1,x=BHAVANA1.index,y=cat_columns[i*(nrows-2)+j],palette='rocket',ax=ax[i][j])
            without_hue(BHAVANA,cat_columns[i-1],a1)
            #Dealing with spines
            ax[i][j].spines['top'].set_visible(False)
            ax[i][j].spines['right'].set_visible(False)
            ax[i][j].spines['left'].set_visible(False)
            ax[i][j].grid(linestyle="--",axis='y',color='gray')


# In[12]:


sns.set_theme(style='white',context='notebook')

fig=plt.figure(figsize=(16,8))

ax=plt.axes()
ax.set_facecolor("#F2EDD7FF")
fig.patch.set_facecolor("#F2EDD7FF")

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(linestyle="--",axis="y",color='gray')

lower_year=2001
upper_year=2014
arr=[]
for i in range(lower_year,upper_year+1):
    arr.append(i)
arr=np.array(arr)

a=sns.lineplot(data=BHAVANA1,palette='gnuplot_r',linestyle="dashed")#,x=arr,y='Rape',hue_order=crimes,palette='rocket_r')

ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
#plt.text(0.5,540000,"Crime rate against women increases year by year",fontweight='bold',fontsize=20)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel("Year",fontweight='bold')
plt.ylabel("crime rate",fontweight='bold')
plt.title("Crime Rate against Women in India",fontweight='bold',fontsize=20)
plt.show()


# # 2.Gender Violence Data: Violence Against Women and Girls ACROSS WORLD.

# In[13]:


import pandas  as pd
import seaborn as sns
import plotly.express    as px
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


# In[14]:


# PRE PROCESSING DATA
BHAVANA = pd.read_csv('violence_data.csv')
BHAVANA


# In[15]:


BHAVANA.isnull().sum()


# In[16]:


print('Dataset contains data from {} countries'.format(BHAVANA.Country.nunique()))


# In[17]:


BHAVANA1 = BHAVANA.pivot_table(index=['Country','Gender','Demographics Question','Demographics Response'],columns=['Question'], values=['Value'])
BHAVANA1


# In[18]:


# Reset columns
survey_df = BHAVANA1.T.reset_index(drop=True).T.reset_index()

# Rename columns
survey_df.columns = ['country',
                    'gender',
                    'demographics_question',
                    'demographics_response',
                    'violence_any_reason',
                    'violence_argue',
                    'violence_food',
                    'violence_goingout',
                    'violence_neglect',
                    'violence_sex',
                   ]


# In[19]:


survey_df


# In[20]:


# Examine Violence x gender
fig = px.box(survey_df.query("demographics_question == 'Age'").sort_values('violence_any_reason',ascending=False),
            x      = 'country',
            y      = 'violence_any_reason',
            color  = 'gender',
            title  = '% of Respondents that agree with violence for any surveyed reason across Country and Gender',
            color_discrete_sequence = ['#4a00ba','#00ba82'],
            height = 650
        )

fig.update_xaxes(title='Country')
fig.update_yaxes(title='% Agrees: Violence is justified for any surveyed reason')
fig.show()


# In[21]:


# Examine Violence x Age group
fig = px.bar(survey_df.query("demographics_question == 'Age'").sort_values('violence_any_reason',ascending=False),
            x      = 'country',
            y      = 'violence_any_reason',
            color = 'demographics_response',
            title  = '% of Violence for any surveyed reason across Country and Age Group ',
            height = 650
        )

fig.update_xaxes(title='Country')
fig.update_yaxes(title='% Agrees: Violence is justified for any surveyed reason')
fig.show()


# In[22]:


# Examine Correlations
plt.figure(figsize=(10,10))
sns.heatmap(survey_df.iloc[:,4:].corr(),
            square=True,
            linewidths=.5,
            cmap=sns.diverging_palette(10, 220, sep=80, n=7),
            annot=True,
           )
plt.title('Correlation Across Different Violence Questions')
plt.show()


# In[ ]:




