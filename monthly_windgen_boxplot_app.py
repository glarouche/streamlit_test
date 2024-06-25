import numpy as np
from coolapp import sesco_wind_gen_state_forecast_damc, three_tier_wind_gen_actuals, three_tier_wind_farms, sesco_wind_gen_forecast
import datetime as dt
import pytz
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import *; from dateutil.relativedelta import *
import streamlit as st

def db_to_df(data, fcst=False):
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['datetime'])
    df = df.set_index('time')
    df['day'] = pd.DatetimeIndex(df.index).day
    df['hour'] = pd.DatetimeIndex(df.index).hour
    df['month'] = pd.DatetimeIndex(df.index).month
    df['week'] = pd.Index(df.index.isocalendar().week, dtype=np.int64)
    df['year'] = pd.DatetimeIndex(df.index).year
    df['date'] = df.datetime.dt.strftime('%m-%d')

    if fcst:
        df['time'] = pd.to_datetime(df['datetime'])
        df['day'] = pd.DatetimeIndex(df['time']).day
        df['hour'] = pd.DatetimeIndex(df['time']).hour
        df['month'] = pd.DatetimeIndex(df['time']).month
        df['week'] = pd.Index(df.index.isocalendar().week, dtype=np.int64)
        df['year'] = pd.DatetimeIndex(df['time']).year
    return df
    
# Read in 3tier metadata 
farm_df = three_tier_wind_farms()
# Set a list of states abbreviations and a state map for all states in the three tier actuals database
state_abr = ['WI', 'NC', 'MT', 'SD', 'MO', 'NM', 'PA', 'NJ', 'TX', 'OK', 'OH', 'VA', 'ND', 'IL', 'IN', 'WV', 'MN', 'NE', 'MD', 'MI', 'KS', 'IA']
state_map = {'WI':'Wisconsin', 'NC': 'North Carolina', 'MT': 'Montana', 'SD': 'South Dakota', 'MO': 'Missouri', 'NM': 'New Mexico', 'PA': 'Pennsylvania', 'NJ':'New Jersey', 'TX': 'Texas', 'OK':'Oklahoma', 'OH':'Ohio', 'VA':'Virginia', 'ND':'North Dakota', 'IL':'Illinois', 'IN':'Indiana', 'WV':'West Virginia', 'MN': 'Minnesota', 'NE':'Nebraska', 'MD':'Maryland', 'MI':'Michigan', 'KS':'Kansas', 'IA':'Iowa'}


st.set_page_config(layout="wide")
st.title('Test Streamlit App Dev')

# Set begin/end time for reading in 3tier actuals
# set time zone as US/Eastern
est = pytz.timezone('US/Eastern')
# Define current time and localize to est
now = est.localize(dt.datetime.now())

# Open a select menu 
cols = st.columns(5)
states = cols[0].multiselect("Select State", list(state_map.values()), default = ['Texas', 'Iowa'])

# Set the historical estimated 3tier actuals period 
hist_year_st = cols[1].selectbox("Select historical start year", list(range(2020,now.year)))
hist_year_end = cols[2].selectbox("Select historical end year", list(range(2023,now.year)))
### check if the historical stuff can just select states right from the beginning, like it can with iso

#Set date range to download the historical values 
start_hist = now.replace(year = hist_year_st, month = 1, day=1, hour = 0, minute=0, second=0, microsecond=0)
end_hist = now.replace(year = hist_year_end, month = now.month+1, day=1, hour = 0, minute=0, second=0, microsecond=0)


# Set the period of actuals plotted on top of boxplots
## Maybe just have it get most recent 30 day period. Need to make it all connected and intuitive with the whole 30 year period. 
## is there a way to only download that month of data? For now, masking entire 4 year chunk is fine but.. wait chunk doesnt have recent stuff 
# for now, lets just select an entire month
months = cols[3].selectbox("Select Month(s)", list(range(1,13)))


# Set masks to load data in range 
begin= now.replace(month = months, day=1, hour = 0, minute=0, second=0, microsecond=0)
end = now.replace(month = months+1,day=1, hour = 23, minute=0, second=0, microsecond=0)

# Set masks to filter historical data by day 
begin_act = begin.strftime('%m-%d')
end_act = end.strftime('%m-%d')
# Read in actuals
actual_3t_df = three_tier_wind_gen_actuals(start_hist, end_hist, iso=['miso', 'spp', 'pjm'])
# joined the 3tier dataset to the farm metadata, so we can group wind gen by state
joined = actual_3t_df.set_index('three_tier_farm_name').join(farm_df.set_index('three_tier_farm_name'), lsuffix='_3t')
df = joined[~joined.state.isna()]

df_hist = db_to_df(df)
df_hist.loc[:,'pct_mw'] = df_hist.mw / df_hist.max_capacity

df_hist_month = df_hist[df_hist['month'] == months]
df_hist_stmonth = df_hist_month[df_hist_month.state.isin(states)]

# Use a pivot table to sum mw by state for each hour 
dfsumm = df_hist_stmonth.pivot_table(index=['datetime', 'hour', 'state'], values= ['mw', 'pct_mw'], aggfunc={'mw':'sum', 'pct_mw': 'mean'})
st_group_m = dfsumm.groupby('state')

# Create a figure and axes for subplots
fig, axes = plt.subplots(len(states), 1, dpi=600, figsize=(10,20), sharey=False)
fig.suptitle(f'Past 3-year Hourly Wind Generation: June', y=1, fontsize = 20)
# set xaxis for obs as 0-23 hours
obs_time = np.arange(24)
# Iterate through each subplot and plot each state's boxplot series
for (ax, (state, state_data)) in (zip(axes.ravel(), st_group_m)):

    
    sns.boxplot(data=state_data, x='hour', y='mw', ax=ax, color = 'lightgray')
    '''
    n = 30
    cmap = plt.cm.jet(np.linspace(0,2,n))
    df_obs= df1[df1['state']==state]
    for i in range(1, df1.day.max()):
    

        df_obs_plot = df_obs[df_obs['day'] == i]
    
        ax.plot(obs_time, df_obs_plot.mw.values, color=cmap[i], linestyle='-', linewidth = 2, label=f'{now.month}/{i}')
    '''
    
    ax.legend().set_visible(False)
    ax.set_title(f'{state}')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Wind Production (mW)')
    ax.grid(True, linestyle='--', alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 0.97),fancybox=True, shadow=False, loc='right', borderpad=.2)
plt.tight_layout()
st.pyplot(fig)
plt.show()
