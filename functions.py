# Cleaning 









# Beuatiful Soup Function for International and Domestic Cities
def get_cities(dom_cities , int_cities):
    
    pops = []
    """Iterate Through website and give me all population of US cities"""
    for city in dom_cities:
        try:
            page1 = requests.get(f'http://worldpopulationreview.com/us-cities/{city}-population/')
            soup1 = BS(page1.content, 'html.parser')
            data = soup1.find(class_ = 'table-striped').get_text().split('%')
            for y in data[:10]:   
                town = {}
                town['city'] = city
                town['year'] = y.split()[-1][:4]
                town['population'] = re.match('^(\d{1,3}(,\d{3})*)?', y.split()[-1][4:]).group(0)
                print(f'Added {town}')
                pops.append(town)
        except:
            print(f"Couldn't retrieve {city}")
    
    """Iteration for population index of International Cities"""
    for city in int_cities:
        try:
            page1 = requests.get(f'http://worldpopulationreview.com/world-cities/{city}-population/')
            soup1 = BS(page1.content, 'html.parser')
            data = soup1.find(class_ = 'table-striped').get_text().split('%')
            for y in data[2:10]:   
                town = {}
                town['city'] = city
                new = re.sub('^(\d{1,3}(,\d{3})*)?', '', y.split()[-1])
                town['year'] = new.split()[-1][:4]
                town['population'] = re.match('^(\d{1,3}(,\d{3})*)?', new.split()[-1][4:]).group(0)
                print(f'Added {town}')
                pops.append(town)
        except:
            print(f"Couldn't retrieve {city}")


# WeatherGetter 
def weatherGetter(stations):
    """Create a dictionary for historical weather for cities"""
    city_weather_history = []
    city = {}
    
    """Iterate Through Response"""
    for i, v in stations.items():
        response = requests.get(f'https://api.meteostat.net/v1/history/monthly?station={v}&start=2009-01&end=2019-01&key=exbuNW5R').json() 
        for m in response['data']:
            try:
                """Create a new DataFrame for the ff columns am interested in """
                month = {}
                month['city_name'] = i
                month['month'] = m['month']
                month['temperature_mean'] = m['temperature_mean']
                month['precipitation'] = m['precipitation']
                month['rain_days'] = m['raindays']
                month['pressure'] = m['pressure']

                city_weather_history.append(month) 
                """print(f"Retrieved {i}'s weather data")"""
            except:
                month['city_name'] = i
                month['month'] = 0
                month['temperature_mean'] = 0
                month['precipitation'] = 0
                month['rain_days'] = 0
                month['pressure'] = 0
                city_weather_history.append(month) 
                 """print(f"Could not retrieve {i}'s weather data")"""
    
    return city_weather_history



# Stationarity Check
def stationarity_check(df1):
    
    """Import adfuller"""
    from statsmodels.tsa.stattools import adfuller
    
    """Calculate rolling statistics"""
    rolmean = df.rolling(window = 8, center = False).mean()
    rolstd = df.rolling(window = 8, center = False).std()
    
    """Perform the Dickey Fuller Test"""
    dftest = adfuller(df1['passengers']) # change the passengers column as required 
    
    """Plot rolling statistics:"""
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(df, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    """Print Dickey-Fuller test results"""
    print ('Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
    return None


def fit_predict_model(dataframe, interval_width = 0.95, changepoint_range = 0.8):
    """Instantiate fbprophet here"""
    m = pr(daily_seasonality = False, yearly_seasonality = True, weekly_seasonality = False,
                seasonality_mode = 'multiplicative', 
                mcmc_samples=1000,
                interval_width = interval_width,
                changepoint_range = changepoint_range)
    """Fit the prophet model to the DataFrame"""
    m = m.fit(dataframe)
    """Forecast prediction"""
    forecast = m.predict(dataframe)
    forecast['fact'] = dataframe['y'].reset_index(drop = True)
    return forecast


# Detecting Anomalities
def detect_anomalies(forecast):
    """Forecasting and make a copy of forecasts"""
    forecasted = forecast[['ds','trend','yhat','yhat_lower','yhat_upper','fact']].copy()
    #forecast['fact'] = df1['y']
    
    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'],'anomaly'] =1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'],'anomaly'] = -1
    
    """Anomality importance"""
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] == 1, 'importance'] = \
         (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] == -1 , 'importance'] = \
         (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']
    
    return forecasted


# Graph Anomalities
def graph_anomalies(city, df1):
    df1 = df1[df1['city'] == city]
    df1 = df1.rename(columns={'PASSENGERS': 'y'})
    df1.head()
    
    """Instantiate facebook prophet here to make predictions"""
    m = pr(daily_seasonality = False, yearly_seasonality = True, weekly_seasonality = False,
                seasonality_mode = 'multiplicative', 
                interval_width = 0.90,
                changepoint_range = 0.8)
    
    """Fit the fbprophet to dataframe"""
    m = m.fit(df1)
    forecast = m.predict(df1)
    forecast['fact'] = df1['y'].reset_index(drop = True)
    
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    #forecast['fact'] = df['y']

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    """anomaly importances"""
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper'])/forecasted['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact'])/forecasted['fact']
    
    """Plot scatter plots"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecasted['ds'], y=forecasted['fact'],
        line_color='rgb(0,100,80)',
        name='Actual'))

    fig.add_trace(go.Scatter(
        x=forecasted['ds'],
        y=forecasted['yhat_upper'],
        fill="tonexty",
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Predicted Upper'))

    fig.add_trace(go.Scatter(
        x=forecasted['ds'],
        y=forecasted['yhat_lower'],
        fill="tonexty",
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Predicted Lower'))

    size=[]
    [size.append(20) for i in range(len(forecasted[forecasted['anomaly'] != 0]))]
    color = []
    [color.append(1) for i in range(len(forecasted[forecasted['anomaly'] != 0]))]

    fig.add_trace(go.Scatter(x=forecasted[forecasted['anomaly'] != 0]['ds'],
                             y=forecasted[forecasted['anomaly'] != 0]['fact'], 
                             name="Anomaly", 
                             mode='markers',
                             marker=dict(size=size,
                                        color=color)))

    fig.update_layout(title=f'{city} Passenger Volume',
                      yaxis_zeroline=False, xaxis_zeroline=False)

    fig.show() 
    
    rmse = sqrt(mean_squared_error(forecasted['fact'], forecasted['yhat']))
    print(rmse)
    
    fig.write_image('iceland_anomalies.jpeg', scale=2)
    

# Future Sarima Model Prediction
def fit_predict_sarima_future(df1, city):
    """Subset for city column in dataframe"""
    df1 = df1[df1['city'] == city]
    df1 = df1.rename(columns={'PASSENGERS': 'y'}).set_index('ds')
    
    """Create Sarimax Model and fit the model"""
    model = SARIMAX(df1['y'],order=(0, 1, 0),seasonal_order=(0,1,0,12))
    results = model.fit()
    
#     print(results.summary())
    """Predicitions"""
    predictions = results.get_forecast(steps=36)
    
    pred_for = predictions.predicted_mean
    spred = pred_for.to_frame()
    
#     predictions = results.predict(dynamic=False, typ='levels')
#     predictions = pd.DataFrame(predictions).reset_index().rename(columns={'index': 'ds', 0: 'y'})
    # Plot predictions against known values
    
    """Plot Predicitons Against Known Values"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1.reset_index()['ds'], 
                             y=df1['y'], 
                             mode='lines',
                             name='Actual',
                             line_color = 'rgb(0, 48, 143)'))

    fig.add_trace(go.Scatter(x=spred.index, 
                             y=spred[0], 
                             mode='lines',
                             name='Predictions',
                             line_color = 'rgb(200,0,100)'))
    
    fig.update_layout(title = f'Future Predictions for {city}', xaxis_title='Year',
                  yaxis_title='Number of Passengers')
    
    fig.write_image('future_image.jpeg', scale=2)
    
    return fig    


# Dickey Fuller Test
def stationarity_check(df1):
    
    """Import adfuller"""
    from statsmodels.tsa.stattools import adfuller
    
    """Calculate rolling statistics"""
    rolmean = df1.rolling(window = 8, center = False).mean()
    rolstd = df1.rolling(window = 8, center = False).std()
    
    '''Perform the Dickey Fuller Test'''
    dftest = adfuller(df1['PASSENGERS']) # change the passengers column as required 
    
    '''Plot rolling statistics:'''
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(df1, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    '''Print Dickey-Fuller test results'''
    print ('Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
    return None


# Plot Anomalies
def plot_anomalities(df1 , ds):
    """Plot Using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
    x=pred['ds'], y=pred['fact'],
    line_color='rgb(0,100,80)',
    name='Actual'))

    fig.add_trace(go.Scatter(
    x=pred['ds'],
    y=pred['yhat_upper'],
    fill="tonexty",
    fillcolor='rgba(0,100,80,0.2)',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
    name='Predicted Upper'))

    fig.add_trace(go.Scatter(
    x=pred['ds'],
    y=pred['yhat_lower'],
    fill="tonexty",
    fillcolor='rgba(0,100,80,0.2)',
    line_color='rgba(255,255,255,0)',
    showlegend=False,
    name='Predicted Lower'))

    size=[]
    [size.append(20) for i in range(len(pred[pred['anomaly'] != 0]))]
    color = []
    [color.append(1) for i in range(len(pred[pred['anomaly'] != 0]))]

    fig.add_trace(go.Scatter(x=pred[pred['anomaly'] != 0]['ds'],
                         y=pred[pred['anomaly'] != 0]['fact'], 
                         name="Anomaly", 
                         mode='markers',
                         marker=dict(size=size,
                                    color=color)))

    fig.update_layout(title='Croatia Passenger Volume',
                  yaxis_zeroline=False, xaxis_zeroline=False)

    fig.show() 
    
    
# Plot Exogenous for Sarimax Prediction
def fit_predict_sarimax_exog(df1, city):
    """Subset DF for City Column"""
    df1 = df1[df1['city'] == city]
    df1 = df1.rename(columns={'PASSENGERS': 'y'}).set_index('ds')
    
    """Train and Test For Endo and Exog Variables"""
    size = int(len(df1) * 0.7)
    train, test = df1['y'][0:size], df1['y'][size:len(df1)]
    train_exog, test_exog = df1[0:size], df1[size:len(df1)]
    
    """Endogenous And Exogenous Train"""
    endog = train_exog['y']
    exog = train_exog[['temperature_mean']]

    """Fit the model"""
    mod = sm.tsa.statespace.SARIMAX(endog, exog, order=(1,0,0,0,1))
    res = mod.fit(disp=False)
    print(res.summary())
    
    start=len(train)
    end=len(train)+len(test)-1

    exog_forecast = test_exog[['temperature_mean']]
    
    """Predicitions"""
    predictions = res.predict(start=start, end=end, exog=exog_forecast, dynamic=False, typ='levels')
    predictions = pd.DataFrame(predictions).reset_index().rename(columns={'index': 'ds', 0: 'y'})
#     predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')
#     predictions = pd.DataFrame(predictions).reset_index().rename(columns={'index': 'ds', 0: 'y'})
    
    """Plot predictions against known values"""

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df1.reset_index()['ds'], 
                             y=df1['y'], 
                             mode='lines',
                             name='Actual',
                             line_color = 'rgb(0, 48, 143)'))

    fig.add_trace(go.Scatter(x=predictions['ds'], 
                             y=predictions['y'], 
                             mode='lines',
                             name='Predictions',
                             line_color = 'rgb(200,0,100)'))
    
    """Calculate Root Mean Square"""
    rms = sqrt(mean_squared_error(test, predictions['y']))
    print(rms)
    
    return fig


# Predictions for FbProphet
def fit_predict_prophet(df1, city):
    """Subset for city column and rename Passengers columns"""
    df1 = df1[df1['city'] == city]
    df1 = df1.rename(columns={'PASSENGERS': 'y'})
    
    size = int(len(df1) * 0.7)
    train, test = df1[['ds','y']][0:size], df1[['ds','y']][size:len(df1)]
#     train_exog, test_exog = df1[0:size], df1[size:len(df1)]
    
    """Reset Index for both training and testing set"""
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    
    start=len(train)
    end=len(train)+len(test)-1
    
#     endog = train_exog['y']
#     exog = train_exog[['temperature_mean']]
    
    """Prophet Modeling and Fit"""
    Model = pr(daily_seasonality = False, yearly_seasonality = True, weekly_seasonality = False,
                seasonality_mode = 'multiplicative', 
                interval_width = 0.90,
                changepoint_range = 0.8)
    Model.fit(train)
    
    """Forecast and Plot Model"""
    forecast = Model.predict(pd.DataFrame(df1.reset_index()['ds'][start-1:end]))
    forecast.head()
    Model.plot(forecast, uncertainty=True)
    
    """Plot Fbprophet Using plotly"""
    from fbprophet.plot import plot_plotly
    import plotly.offline as py
    py.init_notebook_mode()
    
    """This returns a plotly Figure"""
    fig = plot_plotly(Model, forecast)  
    
    """Calculate Root Mean Squared"""
    rms = sqrt(mean_squared_error(test['y'], forecast.loc[:, 'yhat']))
    print(rms)

    return fig

#     py.iplot(fig)
    

# Future Predictions of Prophet
def fit_predict_prophet_future(df1, city):
    df1 = df1[df1['city'] == city]
    df1 = df1.rename(columns={'PASSENGERS': 'y'})
    
    size = int(len(df1) * 0.7)
    train, test = df1['y'][0:size], df1['y'][size:len(df1)]
#     train_exog, test_exog = df[0:size], df[size:len(df)]
    
    """Endog and exog of Passengers and Temperature Mean"""
    endog = train_exog['y']
    exog = train_exog[['temperature_mean']]
    
    """Instantiate Prophet Model and Fit here"""
    Model = pr(interval_width=0.90)
    Model.fit(df1)
    
    """Forecast and prediction"""
    forecast = Model.predict(pd.DataFrame(df1.reset_index()['ds'][start:end]))
    forecast.head()
    Model.plot(forecast, uncertainty=True)
    plt.show()
    
    """USe make_future_dataframe with a monthly frequency and periods = 36 for 3 years"""
    future_dates = Model.make_future_dataframe(periods=36, freq='MS')
    
    """Forecast"""
    forecast = Model.predict(future_dates)














































































