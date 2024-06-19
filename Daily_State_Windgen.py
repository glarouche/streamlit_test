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
from matplotlib.backends.backend_pdf import PdfPages

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
    # try to clean the data. Should not be zero degrees in summer
    #df[df['temperature'] <= 32] = np.nan
    #df[df['temperature'] >= 120] = np.nan
#    df[df['dew'] == 0] = np.nan
#    df[df['dew'] >= 100] = np.nan
    
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
# Check function? if it becomes an issue 

# Set a list of states abbreviations and a state map for all states in the three tier actuals database
state_abr = ['WI', 'NC', 'MT', 'SD', 'MO', 'NM', 'PA', 'NJ', 'TX', 'OK', 'OH', 'VA', 'ND', 'IL', 'IN', 'WV', 'MN', 'NE', 'MD', 'MI', 'KS', 'IA']
state_map = {'WI':'Wisconsin', 'NC': 'North Carolina', 'MT': 'Montana', 'SD': 'South Dakota', 'MO': 'Missouri', 'NM': 'New Mexico', 'PA': 'Pennsylvania', 'NJ':'New Jersey', 'TX': 'Texas', 'OK':'Oklahoma', 'OH':'Ohio', 'VA':'Virginia', 'ND':'North Dakota', 'IL':'Illinois', 'IN':'Indiana', 'WV':'West Virginia', 'MN': 'Minnesota', 'NE':'Nebraska', 'MD':'Maryland', 'MI':'Michigan', 'KS':'Kansas', 'IA':'Iowa'}

# Set begin/end time for reading in 3tier actuals and sesco forecast data
# set time zone as US/Eastern
est = pytz.timezone('US/Eastern')
# Define current time and localize to est
now = est.localize(dt.datetime.now())
# Set lower bound to begin data 4 days ago until tomorrow
start_fcst = now + relativedelta(days=-2)
end_fcst = now+relativedelta(days=2)

# Read in sesco wind gen forecast
state_fcst_df = sesco_wind_gen_state_forecast_damc(start_fcst, end_fcst, market='spp', iso=['miso', 'spp', 'pjm'])
df_fcst = db_to_df(state_fcst_df, True)

# Replace state abbreviations with state names for consistency with estimated actual data 
df_fcst['state'] = df_fcst['state'].replace(state_map)
df_fcst = df_fcst[df_fcst['model_type'] == 'SESCO']

# Select specific forecast initiations for plotting, the hour=8 runs. We hard code BalDay and tomorrow's forecast. 
init = now.replace(hour = 8, minute=0, second=0, microsecond=0)-timedelta(1)
init_ahead = now.replace(hour = 8, minute=0, second=0, microsecond=0)
fcst_balday = df_fcst[(df_fcst['forecasted_at'] == init.strftime('%Y-%m-%d %H:%M:%S'))]
fcst_ahead = df_fcst[(df_fcst['forecasted_at'] == init_ahead.strftime('%Y-%m-%d %H:%M:%S'))]

# Read in 3tier estimated actuals from past 2 days. Use start fcst and now 
actual_3t_3day = three_tier_wind_gen_actuals(start_fcst, now, iso=['miso', 'spp', 'pjm'])

# joined the 3tier dataset to the farm metadata, so we can group wind gen by state
joined = actual_3t_3day.set_index('three_tier_farm_name').join(farm_df.set_index('three_tier_farm_name'), lsuffix='_3t')
df_3day = joined[~joined.state.isna()]
# Extract states for later iterations
states = list(set(joined.state.values))

df_3tier = db_to_df(df_3day)

# Set yesterdays date for obs
yest_obs = now.replace(hour = 8, minute=0, second=0, microsecond=0)-timedelta(1)
# Get historical data for previous day actuals 
obs = df_3tier[(df_3tier['date'] == yest_obs.strftime('%m-%d'))]


# Read in past 3 years of data starting 11 months ago, and running back to 3 years and one month ago. This time is hard coded in but could easily be a selected option
start_hist = (now+relativedelta(years=-3, days = -30)).replace(hour = 0, minute=0, second=0, microsecond=0)
end_hist = (now +relativedelta(months=-11)).replace(hour = 0, minute=0, second=0, microsecond=0)

# Read in actuals
actual_3t_df = three_tier_wind_gen_actuals(start_hist, end_hist, iso=['miso', 'spp', 'pjm'])
# joined the 3tier dataset to the farm metadata, so we can group wind gen by state
joined = actual_3t_df.set_index('three_tier_farm_name').join(farm_df.set_index('three_tier_farm_name'), lsuffix='_3t')
df = joined[~joined.state.isna()]


df.loc[:,'pct_mw'] = df.mw / df.max_capacity
df_hist = db_to_df(df)

# Create time ranges for masking purposes so we can have the previous 30 days of data 

end_mask = now.strftime('%m-%d')

start = now + relativedelta(days = -30)
start_mask = start.strftime('%m-%d')

mask = (df_hist['date'] > start_mask) & (df_hist['date'] <= end_mask)

df_hist = df_hist.loc[mask]

# Date for masking purposes of forecast days

fcst_mask_ahead = now+timedelta(1)
fcst_mask_ahead = fcst_mask_ahead.strftime('%m-%d')
fcst_ahead=fcst_ahead[fcst_ahead['date'] == fcst_mask_ahead] 

fcst_balday=fcst_balday[fcst_balday['date'] == end_mask]


# Use a pivot table to sum mw by state for each hour 
dfsum = df_hist.pivot_table(index=['datetime', 'hour', 'state'], values= ['pct_mw', 'mw'], aggfunc={'mw':'sum', 'pct_mw':'mean'})
st_group = dfsum.groupby('state')


# Create a figure and axes for subplots
fig, axes = plt.subplots(11, 2, dpi=600, figsize=(20, 40), sharey=False)
fig.suptitle(f'Past 3-year Hourly Wind Generation from {start_mask} - {end_mask}', y=1, fontsize = 20)
# set xaxis for obs as 0-23 hours
obs_time = np.arange(24)
# Iterate through each subplot and plot each state's boxplot series
for (ax, (state, state_data)) in (zip(axes.ravel(), st_group)):

    # Subset balday and day ahead forecasts, and yesterday's obs, for plotting along boxplots
    fcst_state_ahead = list(fcst_ahead[fcst_ahead['state'] == str(state)].groupby(by=['datetime']).mw.sum().values)
    fcst_state_balday = list(fcst_balday[fcst_balday['state'] == str(state)].groupby(by=['datetime']).mw.sum().values)

    #Plot time axis without having to hard code it in, in case more hours of a forecast are cut off etc. Would like to eliminate these lines.  
    time_n = fcst_ahead[fcst_ahead['state'] == str(state)].groupby(by=['datetime']).mw.sum().index
    time_y = fcst_balday[fcst_balday['state'] == str(state)].groupby(by=['datetime']).mw.sum().index
    time_ahead = list(time_n.hour)
    time_balday = list(time_y.hour)
    
    obs_plot = obs[obs['state']==str(state)].groupby('hour').mw.sum()
    
    sns.boxplot(data=state_data, x='hour', y='mw', ax=ax, color = 'lightgray')
    

    ax.plot(obs_time, obs_plot, color='red', linestyle='-', label=f'{(init).date()} (Est. Actuals)')
    ax.plot(time_balday, fcst_state_balday, color='blue', linestyle='-', label=f'{(init+timedelta(1)).date()}')
    ax.plot(time_ahead, fcst_state_ahead, color='green', linestyle='-', label=f'{(init+timedelta(2)).date()}')
    ax.set_title(f'{state}')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Wind Production (mW)')
    # Set x-axis ticks to show hours from 0 to 23
    ax.set_xticks(obs_time)
    # Add legend
    ax.legend(loc='upper left', fontsize='small', prop={'size':8})
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'wind_gen/pdf_boxplots_windgen_{now.date()}.pdf', format='pdf')
plt.show()









