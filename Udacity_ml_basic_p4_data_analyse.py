
# coding: utf-8

# # This is Udacity ML basic P4 heading line

# ### 1.1 read the csv file and check & clean the row data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'pylab inline')
G_CHILD_AGE = 18
G_UNKNOWN_AGE = -1
tt_file = pd.read_csv("titanic_data.csv", index_col = "PassengerId")
tt_file.head()

###### 1.2 Raw data field info
 $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
 $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
 $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
 $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
 $ Sex        : chr  "male" "female" "female" "female" ...
 $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
 $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
 $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
 $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
 $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
 $ Cabin      : chr  "" "C85" "" "C123" ...
 $ Embarked   : chr  "S" "C" "S" "S" ...

Variable Name 	Description
Survived 	Survived (1) or died (0)
Pclass 	Passenger’s class
Name 	Passenger’s name
Sex 	Passenger’s sex
Age 	Passenger’s age
SibSp 	Number of siblings/spouses aboard
Parch 	Number of parents/children aboard
Ticket 	Ticket number
Fare 	Fare
Cabin 	Cabin
Embarked 	Port of embarkation
# ### 2.1 Check overall the survived VS non-survived people

# In[2]:



def convert_child_age(df):
    if df.Age < G_CHILD_AGE:
        df.Sex =  'child'
    return df

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/3, 1.03*height, '%d' % int(height))


# In[3]:


survived_overview = tt_file.apply(convert_child_age, axis=1).groupby(['Survived',"Sex"])['Sex'].count()
print survived_overview
##type(survived_overview)

non_survived_C = survived_overview.iloc[0]
non_survived_F = survived_overview.iloc[1]
non_survived_M = survived_overview.iloc[2]
survived_C = survived_overview.iloc[3]
survived_F = survived_overview.iloc[4]
survived_M = survived_overview.iloc[5]

total_S = survived_F + survived_M + survived_C
total_N = non_survived_F + non_survived_M + non_survived_C
                
y_values = [non_survived_C, non_survived_F, non_survived_M, survived_C, survived_F, survived_M, total_S, total_N]
x_labels = ['NS_Chd', 'NS_Fle','NS_Mle','S_Chd', 'S_Fle','S_Mle','T_Surv', 'T_NS']
bar_width = 0.9

rect1 = plt.bar(x_labels[:3], y_values[:3], label = 'Non Survived', color = 'r', alpha = 0.8, width = bar_width)
rect2 = plt.bar(x_labels[3:6], y_values[3:6], label = 'Survived', color = 'g', alpha = 0.8, width = bar_width)
rect3 = plt.bar(x_labels[6:8], y_values[6:8], label = 'Total', color = 'k', alpha = 0.8, width = bar_width)

plt.xlabel('Types Of Statistics')
plt.ylabel('Numbers Of People')
plt.title('Overview\nSurived Vs Non-surived by Sex')

#plt.xticks(np.arange(7) + bar_width, x_labels)

plt.ylim([0, 600])
    
plt.legend()
autolabel(rect1)
autolabel(rect2)
autolabel(rect3)

plt.show()

male_ratio = [round(float(non_survived_M) / (total_S + total_N), 2), 
              round(float(non_survived_F) / (total_S + total_N), 2),
              round(float(non_survived_C) / (total_S + total_N), 2),
              round(float(survived_M) / (total_S + total_N), 2),
              round(float(survived_F) / (total_S + total_N), 2),
             round(float(survived_C) / (total_S + total_N), 2)]
print "overall male ration", (float(survived_M) + non_survived_M ) / (total_S + total_N)
print "overall female ration", (float(survived_F) + non_survived_F ) / (total_S + total_N)
male_label = ['non_survived_male', 'non_survived_female','non_survived_child',               'survived_male', 'survived_female', 'survived_child']
male_ratio_pd = pd.Series(male_ratio, index=male_label)
plt.pie(x=male_ratio, labels=male_label, colors=['r', 'm', 'c', 'g','y','g'], autopct='%.2f')

### 2.2 Conclusions from the overview of suurvived people by sex:
1) The ration  of non-survived male people (49%) which implies: 
    -- compare with non-survived female ratio (6.93%) plus non-survied child(5.94)
      -- means most of the male did not survived.
2) The ration of surived female (21.78%) is the second high number with below 2 cases:
    -- compare with non-survived female ratio (6.93%)
      -- means most of the female survived.
3) The ration of surived and non-survived child are almost equivalent.(6.93% Vs 5.94%)  
Notes: the unknown age people are not counted as Child type
# ### 3.1 Consider the age impact from the survived and non-survived people

# In[56]:



#age_overview = tt_file.groupby(['Survived', 'Age']['Age'].describe()
#print age_overview
def pick_non_sur(df):
    if df.Survived == 0 and not np.isnan(df.Age):
        return df.Age
def pick_sur(df):
    if df.Survived == 1 and not np.isnan(df.Age):
        return df.Age

d_n_array = np.array(tt_file.apply(pick_non_sur, axis=1))
d_s_array = np.array(tt_file.apply(pick_sur, axis=1))
#print dn_array

binsize = [0,10,20,30,40,50,60,70,80]
cat_n = pd.cut(d_n_array, binsize)
cat_s = pd.cut(d_s_array, binsize)
#print dn_array
print pd.value_counts(cat_n, dropna=True)
print pd.value_counts(cat_s, dropna=True)
data_age = ((pd.value_counts(cat_s, dropna=True))/ (pd.value_counts(cat_n, dropna=True)                                               + pd.value_counts(cat_s, dropna=True))).sort_values(axis=0,                                               ascending=True)
print data_age

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Non Survived Age Distribution')
ax1.set_xlabel('Age Range')
ax1.set_ylim(0, 140)
ax1.set_yticks([20, 40, 60, 80, 100, 120, 140])
pd.value_counts(cat_n, dropna=True, sort=True).plot(kind='barh')

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('Survived Age Distribution')
ax2.set_yticks([20, 40, 60, 80, 100, 120, 140])
ax2.set_ylim(0, 140)
pd.value_counts(cat_s, dropna=True, sort=True).plot(kind='barh')
ax2.set_xlabel('Age Range')

fig, axe = plt.subplots(1, 1)
data_age.plot(kind='barh', ax=axe, color='k', alpha=0.7,             title='Surived Ration With Compared Age Range', )

### 3.1 Conclusions from the overview of survived people by age:
1) The 0-10 Age (Child) has the most survived ration.
2) The young people (20-40) are the biggest gruop among the survived and non-survived group.### 4.1 Consider the fare impact from the survived and non-survived people
# In[60]:


def pick_non_sur(df):
    if df.Survived == 0 and not np.isnan(df.Age):
        return df.Fare
def pick_sur(df):
    if df.Survived == 1 and not np.isnan(df.Age):
        return df.Fare
    
d_n_array = np.array(tt_file.apply(pick_non_sur, axis=1))
d_s_array = np.array(tt_file.apply(pick_sur, axis=1))
#print dn_array

binsize = [0,10,20,30,40,50,60,70,80,90]
cat_n = pd.cut(d_n_array, binsize)
cat_s = pd.cut(d_s_array, binsize)
#print dn_array
print pd.value_counts(cat_n, dropna=True)
print pd.value_counts(cat_s, dropna=True)
data_fare = ((pd.value_counts(cat_s, dropna=True))/ (pd.value_counts(cat_n, dropna=True)                                               + pd.value_counts(cat_s, dropna=True))).sort_values(axis=0,                                               ascending=True)
print data_fare

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Non Survived Fare Distribution')
ax1.set_xlabel('Fare Range')
ax1.set_ylim(0, 160)
ax1.set_yticks([20, 40, 60, 80, 100, 120, 140])
pd.value_counts(cat_n, dropna=True, sort=True).plot(kind='barh')

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('Survived Fare Distribution')
ax2.set_yticks([20, 40, 60, 80, 100, 120, 140])
ax2.set_ylim(0, 160)
pd.value_counts(cat_s, dropna=True, sort=True).plot(kind='barh')
ax2.set_xlabel('Fare Range')

fig, axe = plt.subplots(1, 1)
data_fare.plot(kind='barh', ax=axe, color='k', alpha=0.7,             title='Surived Ration With Compared Fare Range', )

### 4.2 Conclusions from the overview of survived people by fare:
1) the 0-20 fare range is the most group people on the ship
2) the highest survived raton among the fare range are above 60.### 5.1 Consider the family-szie impact from the survived and non-survived people
# In[58]:


def pick_non_sur(df):
    if df.Survived == 0 and not np.isnan(df.Age):
        return df.SibSp + df.Parch

def pick_sur(df):
    if df.Survived == 1 and not np.isnan(df.Age):
        return df.SibSp + df.Parch
    
d_n_array = np.array(tt_file.apply(pick_non_sur, axis=1))
d_s_array = np.array(tt_file.apply(pick_sur, axis=1))
#print dn_array

binsize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
cat_n = pd.cut(d_n_array, binsize)
cat_s = pd.cut(d_s_array, binsize)
#print dn_array
print pd.value_counts(cat_n, dropna=True)
print pd.value_counts(cat_s, dropna=True)

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('Non Survived Family Distribution')
ax1.set_xlabel('Family Size Range')
ax1.set_ylim(0, 80)
ax1.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
pd.value_counts(cat_n, dropna=True, sort=True).plot(kind='bar')

ax2 = fig.add_subplot(1,2,2)
ax2.set_title('Survived Famiy Distribution')
ax2.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
ax2.set_ylim(0, 80)
pd.value_counts(cat_s, dropna=True, sort=True).plot(kind='bar')
ax2.set_xlabel('Family Size Range')

### 5.2 Conclusions from the overview of survived people by family size:
1) the number of people with the family size = 1 or 2 is the biggest group among both non-survived and survived people.
2) the singletons has the highest chance for survive.
# In[7]:


### 6.1 Consider some field correlations from the survived and non-survived people


# In[51]:



def field_digitization(df):
        if not np.isnan(df.Age):
            if df.Age < 40:
                df.Age = 3
            elif df.Age >= 40 and df.Age < 60:
                df.Age = 2
            elif df.Age >= 60 and df.Age < 80:
                df.Age = 1
            else:
                df.Age = 0
        if not np.isnan(df.Fare):
            df.Fare = int(round(float(df.Fare) / 10, 0))
        #if not np.isnan(df.Sex):
        if df.Sex == 'male':
            df.Sex = 0
        if df.Sex == 'female':
            df.Sex = 1
        #if not np.isnan(df.Embarked):
        if df.Embarked == 'C':
            df.Embarked = 0
        if df.Embarked == 'Q':
            df.Embarked = 1            
        if df.Embarked == 'S':
            df.Embarked = 2 
        
        return df

base_array = np.array(tt_file['Survived'].values)
corr_index = ['Sex', 'Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Embarked']

tt_refined_file = tt_file.apply(field_digitization, axis=1)

#pick up the integer
result_list = []
for field in corr_index:
    new_array = []
    fd_array = np.array(tt_refined_file[field].values) 
    for i in range(len(fd_array)):
        if not np.isnan(fd_array[i]):
            new_array.append(int(str(fd_array[i]).split('.')[0]))
            #print new_array[i]
        else:
            new_array.append(-1) # default as -1
    rst = round(abs(np.corrcoef(base_array, new_array)[1,0]), 2)
    result_list.append(rst)

fig, axe = plt.subplots(1, 1)
data = pd.Series(result_list, index=corr_index)
print data
data.sort_values(axis=0, ascending=True).plot(kind='barh', ax=axe, color='k', alpha=0.7,             title='Correlation Distribution', )

### 6.1 Some findings
1) the top3 highest correlations fields are 'Sex'', 'Pclass' and 'Fare'.
2) Still confused why Age has such low corrlelation rank than 'Embarked' or 'Fare'. But according to the corrcoef() results, it is not much so high.