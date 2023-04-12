#!/usr/bin/env python
# coding: utf-8

# ## Install required packages

# In[1]:


pip install emoji==1.7.0


# In[2]:


pip install dateparser


# In[3]:


pip install jovian


# ## Import all required packages for the project

# In[4]:


import re


# In[5]:


import matplotlib.pyplot as plot


# In[6]:


import pandas as pd


# In[7]:


import numpy as np


# In[8]:


import seaborn as sns


# In[9]:


import calendar


# In[10]:


import datetime as dt


# In[11]:


from wordcloud import WordCloud,STOPWORDS


# In[12]:


import emoji


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer


# In[14]:


import random


# In[15]:


from textblob import TextBlob


# In[16]:


from sklearn.decomposition import LatentDirichletAllocation


# In[17]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[18]:


from sklearn.decomposition import NMF


# In[19]:


sns.set(rc={'figure.figsize':(14,6)})


# # Data Cleaning

# In[20]:


def startswithDate(s):
    pattern='^([0-2][0-9]|(3)[0-1])(\/)(((0)[0-9])|((1)[0-2]))(\/)(\d{2}|\d{4}), ([0-9][0-9]):([0-9][0-9]) -'
    result=re.match(pattern,s)
    if result:
        return True
    return False


# In[21]:


def startswithAuthor(s):
    patterns=['([\w]+):', # First Name
              '([\w]+[\s]+[\w]+):', # First Name + Last Name
             '([\w]+[\s]+[\w]+[\s]+[\w]+):',  # First Name+ Middle Name + Last Name
             '^(\+\d{1,2}\s)?\(?\d{3}\) ((\(\d{3}\) ?)|(\d{3}-))?\d{3}-\d{4}:',  #Mobile Number(US)
             '([+]\d{2} \d{4} \d{6}):',  # Mobile Number (Europe)
           '([+]\d{3} \d{3} \d{6})'  # Mobile Number (Uganda)
             ]
    pattern='^'+'|'.join(patterns)
    result=re.match(pattern,s)
    if result:
        return True
    return False


# In[22]:


def getDataPoint(line):
    splitLine=line.split(' - ') #splitLine=['18/06/2021, 22:47','Loki: Why do you have two numbers, Arsalan?']
    dateTime=splitLine[0] # '18/06/2021, 22:47'
    message=' '.join(splitLine[1:]) #'Loki: Why do you have two numbers, Arsalan?'
    if startswithAuthor(message): #True
        splitmessage=message.split(': ') #splitmessage=['Loki','Why do you have two numbers, Arsalan?']
        author=splitmessage[0] #author='Loki'
        message=" ".join(splitmessage[1:]) # message='Why do you have two numbers, Arsalan?''
    else:
        author=None
    return dateTime,author,message


# In[23]:


parsedChat=[]
convoPath='./WhatsappChat.txt'
with open(convoPath,encoding='utf-8') as fp:
    fp.readline
    messagebuffer=[]
    dateTime,author=None,None
    
    while True:
        line=fp.readline()
        if not line:
            break
        line=line.strip()
        if startswithDate(line):
            if len(messagebuffer)>0:
                parsedChat.append([dateTime,author," ".join(messagebuffer)])
                messagebuffer.clear()
                dateTime,author,message=getDataPoint(line)
                messagebuffer.append(message)
            else:
                messagebuffer.append(line)
        


# In[24]:


parsedChat


# In[25]:


df=pd.DataFrame(parsedChat,columns=['DateTime','Author','Message'])


# In[26]:


df.head()


# In[27]:


df.tail()


# In[28]:


df.describe()


# In[29]:


# No. of images, images are represented by <Media omitted>
media=df[df['Message']=='<Media omitted>']


# In[30]:


author_media_messages_value_counts=media['Author'].value_counts()
top10_author_media_messages_value_counts=author_media_messages_value_counts.head(10)


# In[31]:


top10_author_media_messages_value_counts.plot.barh()
plot.ylabel('Author')
plot.xlabel('Media items sent')
plot.title('Most Media Items sent per Author')


# In[32]:


message_deleted=df[df['Message']=='This message was deleted']


# In[33]:


message_deleted


# In[34]:


author_messages_deleted_value_counts=message_deleted['Author'].value_counts()


# In[35]:


author_messages_deleted_value_counts


# In[36]:


top10_aithor_messages_deleted_value_counts=author_messages_deleted_value_counts.head(10)


# In[37]:


import matplotlib.pyplot as plt
top10_aithor_messages_deleted_value_counts.plot.barh()
plt.ylabel('Author')
plt.xlabel('Message Deleted')


# In[38]:


#Number of Group Notifications
grp_notif=df[df['Author']=='grp_notif']


# In[39]:


grp_notif.shape


# In[40]:


# Drop the 'media omitted' messages,group notifications and deleted messages
df.drop(media.index,inplace=True)
df.drop(grp_notif.index,inplace=True)
df.drop(message_deleted.index,inplace=True)


# In[41]:


# Find the null values
df.isnull().sum()


# In[42]:


# Drop empty rows
df=df.dropna()


# In[43]:


df.shape


# In[44]:


df.reset_index(inplace=True,drop=True)


# Add the dateTime object from dateTime column

# In[45]:


df['dateTime']=pd.to_datetime(df['DateTime'],infer_datetime_format=True)


# In[46]:


df['Day of Week']=pd.Series(pd.Categorical(df['dateTime'].dt.day_name(),categories=list(calendar.day_name)))


# In[47]:


df['Day of Week']


# In[48]:


df['Hour']=pd.Series(pd.Categorical(df['dateTime'].dt.hour))


# In[49]:


df['Hour']


# In[50]:


df=df.set_index('dateTime')


# In[51]:


df.info()


# In[52]:


df.head()


# ## Data Exploration

# In[53]:


df.describe()


# In[54]:


author_value_counts=df['Author'].value_counts()
top10_talkers=author_value_counts.head(10)


# In[55]:


top10_talkers


# In[56]:


top10_talkers.plot.barh()
plt.xlabel('Number of words')
plt.ylabel('Authors')
plt.title('The top 10 Most Talkatie Persons')


# In[57]:


df


# In[58]:


df['Date']=df['DateTime'].apply(lambda t:t.split(',')[0])


# In[59]:


df['Date']


# In[60]:


df.groupby('Date')['Message'].count().plot()
plt.ylabel('Number of messages')


# In[61]:


df['Time']=df['DateTime'].apply(lambda t: t.split(',')[1])


# In[62]:


df['Time']


# In[63]:


df.groupby('Time')['Message'].count().plot()
plt.ylabel('Number of messages')


# In[64]:


df.groupby('Hour')['Message'].count().plot()
plt.ylabel('Number of messages')


# In[65]:


df['Letter Count']=df['Message'].apply(lambda s: len(s))


# In[66]:


df['Letter Count']


# In[67]:


len('asdas')


# In[68]:


df['Word Count']=df['Message'].apply(lambda s: len(s.split(' ')))


# In[69]:


df['Word Count']


# In[70]:


df['Message']


# In[71]:


df


# In[72]:


print('Total Letter Count in the group:'+str(df['Letter Count'].sum()))


# In[73]:


print('Total Word Count in the group:'+str(df['Word Count'].sum()))


# What is the most common number of words in a message?

# In[74]:


plt.figure(figsize=(20,4))
word_count_value_counts=df['Word Count'].value_counts()
top30_word_count=word_count_value_counts.head(30)
top30_word_count.plot.bar()
plt.xlabel('Word Count')
plt.ylabel('Frequency')


# In[75]:


## Summary by user
users=df.groupby('Author')['Author'].count()


# In[76]:


print(users)


# In[77]:


## Summary by user
msg=df.groupby('Author')['Message'].count()


# In[78]:


print(msg)


# So who exactly writes the most letters?

# In[79]:


total_letter_count_by_author=df[['Author','Letter Count']].groupby('Author').sum()


# In[80]:


sorted_total_letter_count_by_author=total_letter_count_by_author.sort_values('Letter Count',ascending=False)
top10_sorted_total_letter_grouped_by_author=sorted_total_letter_count_by_author.head(10)


# In[81]:


top10_sorted_total_letter_grouped_by_author.plot.barh()
plt.xlabel('Number of letters')
plt.ylabel('Authors')


# In[82]:


total_letter_count_by_author=df[['Author','Letter Count']].groupby('Author').sum()
sorted_total_letter_count_by_author=total_letter_count_by_author.sort_values('Letter Count',ascending=False)
sorted_total_letter_grouped_by_author=sorted_total_letter_count_by_author
sorted_total_letter_grouped_by_author.plot.barh()
plt.xlabel('Number of letters')
plt.ylabel('Authors')


# Most Common number of letter per message

# In[83]:


plt.figure(figsize=(15,2))
letter_count_value_counts=df['Letter Count'].value_counts()
top30_letter_count_value_counts=letter_count_value_counts.head(30)
top30_letter_count_value_counts.plot.bar()
plt.xlabel('Letter Count')
plt.ylabel('Frequency')


# When was the group most active?

# In[84]:


df['Day of Week'].value_counts().plot.barh()
plt.xlabel('Number of Messages')
plt.ylabel('Day of the Week')


# Any Particular dates?

# In[85]:


df['Date'].value_counts().head(10).plot.barh()
plt.xlabel('Number of Messages')
plt.ylabel('Date')


# The most active Hour

# The most suitable time of day to get your message replied to

# In[86]:


df['Hour'].value_counts().head(10).plot.barh() # Top 10 Times of the day at which the most number of messages sent
plt.xlabel('Number of Messages')
plt.ylabel('Hour')


# In[87]:


df['Hour']


# In[88]:


df['Time'].value_counts().head(10).plot.barh() # Top 10 Times of the day at which the most number of messages sent
plt.xlabel('Number of Messages')
plt.ylabel('Time')


# What are the most commonly used words?

# In[89]:


common_words=''
for val in df['Message'].values:
    val=str(val)
    tokens=val.split()
    
    for i in range(len(tokens)):
        tokens[i]=tokens[i].lower()
    
    for words in tokens:
        common_words=common_words + words + ''
famcloud=WordCloud(width=800,height=800).generate(common_words)

plt.figure(figsize=(8,8))
plt.imshow(famcloud)
plt.axis('off')
plt.tight_layout()
plt.show()


# In[90]:


#If you want to download the image, Run the cell
famcloud.to_image()


# In[91]:


df['Author'].unique()


# Find out the group's top 20 emoji usage

# In[92]:


from collections import Counter
emojiGrpCtr=Counter()
emojis_list=map(lambda x: "".join(x.split()),emoji.UNICODE_EMOJI.keys())
r=re.compile('|'.join(re.escape(p) for p in emojis_list))
for idx,row in df.iterrows():
    emojis_found=r.findall(row['Message'])
    for emoji_found in emojis_found:
        emojiGrpCtr[emoji_found]+=1
for item in emojiGrpCtr.most_common(20):
    print(item[0] + ' - '+str(item[1]))


# In[93]:


emojiGrp=pd.DataFrame.from_dict(emojiGrpCtr,orient='index').reset_index()


# In[94]:


emojiGrp=emojiGrp.rename(columns={'index':'Emoji',0:'Count'})


# In[95]:


emojiGrp.head()


# ## Sentiment Analysis

# In[96]:


df['Polarity']=df['Message'].map(lambda text: TextBlob(text).sentiment.polarity)


# Randomly select 5 reviews with the highest positive polarity score

# In[97]:


print('5 random reviews with the highest positive sentiment polarity: \n ')
cl=df.loc[df['Polarity']==1,['Message']].sample(5).values
for c in cl:
    print(c[0])


# Randomly select 5 reviews with the most neutral sentiment polarity score

# In[98]:


print('5 reviews with the most neutral sentiment(zero) polarity: \n')
cl=df.loc[df['Polarity']==0,['Message']].sample(5).values
for c in cl:
    print(c[0])


# In[99]:


print('5 reviews with the most negative polarity: \n')
cl=df.loc[df['Polarity']==-0.50,['Message']].sample(5).values
for c in cl:
    print(c[0])


# The distribution of review sentiment polarity score

# In[100]:


plt.hist('Polarity',data=df)


# ## The End

# In[ ]:




