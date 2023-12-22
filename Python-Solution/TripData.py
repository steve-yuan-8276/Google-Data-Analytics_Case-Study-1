#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# Load csv file
df01 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202201-divvy-tripdata.csv')
df02 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202202-divvy-tripdata.csv')
df03 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202203-divvy-tripdata.csv')
df04 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202204-divvy-tripdata.csv')
df05 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202205-divvy-tripdata.csv')
df06 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202206-divvy-tripdata.csv')
df07 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202207-divvy-tripdata.csv')
df08 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202208-divvy-tripdata.csv')
df09 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202209-divvy-tripdata.csv')
df10 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202210-divvy-tripdata.csv')
df11 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202211-divvy-tripdata.csv')
df12 = pd.read_csv('/Users/steveyuan/Documents/github_files/Google-Data-Analytics_Case-Study-1/original_data/202212-divvy-tripdata.csv')


# In[6]:


print(df01.head)


# In[7]:


print(df01.dtypes)


# In[8]:


print(df01.shape)


# In[9]:


print(df01.columns)


# In[4]:


df = pd.concat([df01, df02, df03, df04, df05, df06, df07, 
                df08, df09, df10, df11, df12], ignore_index= True)
print(df.shape)


# In[8]:


print(df.head)


# ## Data Validation

# In[12]:


# Check NaN value
nan_counts = df.isna().sum()
print(nan_counts)


# In[14]:


# Check if the riding time is abnormal
# check riding time < 0
has_negative_riding_times = (df['riding_time'] < 0).any()    
print('Negative_riding_times:', has_negative_riding_times)


# In[16]:


# check riding time > 1440
has_excessive_riding_times = (df['riding_time'] > 1440).any()    
print('Excessive_riding_times:', has_excessive_riding_times)


# In[17]:


# Remove the data with riding time < 0 or riding time > 1440
df_cleaned = df[(df['riding_time'] >= 0) & (df['riding_time'] <= 1440)]
print(df_cleaned.shape)


# ## Data Analysis

# ### Distribution of Rides by Customer Type

# In[20]:


# calculate the unique customer number
customer_counts = df['customer_type'].value_counts()
print(customer_counts)


# In[24]:


labels = customer_counts.index
sizes = customer_counts.values

# Draw Pie Chart
fig, ax = plt.subplots()
ax.pie(sizes, labels = labels, autopct = '%1.1f%%', startangle = 90)

# Set colors
colors = ['#1f77b4', '#d62728']

# Set Figure Size
plt.figure(figsize=(10, 6))

# Add Legend
ax.legend(labels, title = 'Customer Type')

# Add Title
ax.set_title('Distribution of Rides by Customer Type')

# Show pia chart
plt.show()


# ### Hourly Distribution of Rides by Customer Type

# In[26]:


# count number by hourly and customer type
hourly_customer_counts = df.groupby(['start_hour', 'customer_type']).size().unstack(fill_value= 0)

# sorted hour from 0 to 23
hourly_customer_counts = hourly_customer_counts.sort_index()

hours = hourly_customer_counts.index
casual_counts = hourly_customer_counts['casual']
member_counts = hourly_customer_counts['member']

# Set figure size
plt.figure(figsize = (15, 8))

# Draw bar chart
# Draw member type first
plt.bar(hours, member_counts, label = 'Member', color = '#1f77b4')
# Draw casual type 
plt.bar(hours, casual_counts, bottom= member_counts, label= 'Casual', color = '#d62728')

# Add legend
plt.legend()

# Add Title and Axis labels
plt.title('Hourly Distribution of Rides by Customer Type')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Rides')

plt.xticks(hours)

plt.show()


# ## Weekly Distribution of rides by customer

# In[7]:


# Create a mapping of week days to sort them from Monday to Sunday
week_day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Reorder the DataFrame based on the week_day_order
df['week_day'] = pd.Categorical(df['week_day'], categories=week_day_order, ordered=True)

# count number by week_day and customer type
weekly_customer_counts = df.groupby(['week_day', 'customer_type']).size().unstack(fill_value= 0)

week_days = weekly_customer_counts.index
casual_counts = weekly_customer_counts['casual']
member_counts = weekly_customer_counts['member']

# Set figure size
plt.figure(figsize = (15, 8))

# Draw bar chart
# Draw member type first
plt.bar(week_days, member_counts, label = 'Member', color = '#1f77b4')
# Draw casual type 
plt.bar(week_days, casual_counts, bottom= member_counts, label= 'Casual', color = '#d62728')

# Add legend
plt.legend()

# Add Title and Axis labels
plt.title('weekly Distribution of Rides by Customer Type')
plt.xlabel('Day of Week')
plt.ylabel('Number of Rides')

# Set the position of the x-ticks
plt.xticks(ticks=range(len(week_days)), labels=week_days)

plt.show()


# ## Monthly Distribution of rides by Customer

# In[10]:


# Create a mapping of month to sort them from January to December
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Reorder the DataFrame based on the month_order
df['month'] = pd.Categorical(df['month'], categories=month_order, ordered=True)

# count number by month and customer type
monthly_customer_counts = df.groupby(['month', 'customer_type']).size().unstack(fill_value= 0)

months = monthly_customer_counts.index
casual_counts = monthly_customer_counts['casual']
member_counts = monthly_customer_counts['member']

# Set figure size
plt.figure(figsize = (15, 8))

# Draw bar chart
# Draw member type first
plt.bar(months, member_counts, label = 'Member', color = '#1f77b4')
# Draw casual type 
plt.bar(months, casual_counts, bottom= member_counts, label= 'Casual', color = '#d62728')

# Add legend
plt.legend()

# Add Title and Axis labels
plt.title('monthly Distribution of Rides by Customer Type')
plt.xlabel('Month')
plt.ylabel('Number of Rides')

# Set the position of the x-ticks
plt.xticks(ticks=range(len(months)), labels=months)

plt.show()


# ## Distribution of use by bike type

# In[13]:


# calculate the bike number
bike_counts = df['bike_type'].value_counts()
print(bike_counts)

labels = bike_counts.index
sizes = bike_counts.values

# Draw Pie Chart
fig, ax = plt.subplots()
ax.pie(sizes, labels = labels, autopct = '%1.1f%%', startangle = 90)

# Set colors
colors = ['#1f77b4', '#d62728', '#E69F00']

# Set Figure Size
plt.figure(figsize=(10, 6))

# Add Legend with position set to lower right
ax.legend(labels, title='Bike Type', loc='lower right', bbox_to_anchor=(1, 0))

# Add Title
ax.set_title('Distribution of Rides by Bike Type')

# Show pia chart
plt.show()


# ## Destribution of riding time by customer type

# In[14]:


# calculate the total riding time by each user type
total_ride_time_by_user_type = df.groupby('customer_type')['riding_time'].sum()
#print(bike_counts)

labels = total_ride_time_by_user_type.index
sizes = total_ride_time_by_user_type.values

# Draw Pie Chart
fig, ax = plt.subplots()
ax.pie(sizes, labels = labels, autopct = '%1.1f%%', startangle = 90)

# Set colors
colors = ['#1f77b4', '#d62728']

# Set Figure Size
plt.figure(figsize=(10, 6))

# Add Legend with position set to lower right
ax.legend(labels, title='Customer Type', loc='lower right', bbox_to_anchor=(1, 0))

# Add Title
ax.set_title('Distribution of Riding Time by Customer Type')

# Show pia chart
plt.show()


# ### Data insights
# 
#   - Annual members use the shared bike service more often on weekdays, while casual users use it more frequently on weekends.
#   
#   - There was a clear spike during the off-hours peak (16:00 to 19:00), when annual membership usage was higher.
#   
#   - The effect of seasonal factors on bike-sharing services is evident, with significant peaks in the summer and significant troughs in the winter.
#   
#   - The most popular type of bike is the classic bike, while docked bikes are rarely used and annual members hardly ever use docked bikes.
#   
#   - Casual users have a greater preference for longer rides, with an average ride time 1.7 times longer than annual users.
#   
# ### Recommendations
# 
#   - Increase promotions for regular users, especially during the summer and weekends, to attract them to convert to annual subscribers.
#   
#   - Increase daily vehicle deployment during peak usage periods to further improve user experience.
#   
#   - Further enhance the experience of using classic bicycles and improve user satisfaction.
#   
#   - Further evaluate the reasons for the low usage of docked bicycles and make countermeasures accordingly.

# In[ ]:




