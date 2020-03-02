# Travel_Trends_Project

# World Tourism Trends & Forecasting

## The Question
- Travelling appears to be increasing in popularity, and certain destinations seem to explode overnight. With global economic    factors , rising economies , increase in education which mostly leads to people finding well paid jobs worldwide , thereby increasing spending power across the globe and the increase in social media awareness.

- A few decades ago people were happy to live the normal life of getting a job after school and raising a family cycle ; however , due to the social media boom , people worldwide are much more exposed to what lies beyond their familiar background and are more willing to see and experience the world through the lenses of other travellers. 

- Thereby causing a major boom in travelling across the globe. Most interestingly, some parts of the world which were never considered to be tourist destinations now have major increase in tourism within a very short period and have grown to be very vibrant tourist destinations around the globe due to social media ,online presence , just to mention but a few.

- Does this increase in travel really exist, and can it inform us on future trends? If so , how long wil this continue or is it just a phase to be short lived? 
How can our company target travellers or marketing strategy to increase ticket sales to these upcoming cities and other interesting addons we can upsell to travellers . 
Does the local governments need to invest in infrastructure to cater for tourists?

### Iceland Tourism boom
Tourism grew to one-third of Iceland’s economy in 2015, boosting the country’s dependency on the sector. But things are likely changing, and it’s unclear what comes next.
— Isaac Carey


The tourism boom that saved Iceland from economic crisis is slowing further, according to new tourism numbers released by the Icelandic Tourist Board. Fueled for nearly eight years by international travelers, the tourism industry may be reaching a breaking point, spelling an uncertain future for the Icelandic economy.

In 2018, 2.3 million people visited Iceland, a 5.5 percent increase from the year before, according to data released this month. The Iceland-based media site Túristi first reported the news.

On its own, the year-over-year increase sounds strong, but it points to a slowing trend. While the number of international visitors to Iceland grew by 39 percent in 2016, the increase in 2017 was 24 percent. That was the first time tourism growth slowed since the boom started in 2011.


## The Data
- Webscrapping using Visual Code Editor , 
- Selenium , 
- Beautiful Soup 
- DB Browser(SQL) to store Data and 
- MeteoStat weather API:


## Data Collection 
- transtats.bts.gov : https://www.transtats.bts.gov/DL_SelectFields.asp

- ceicdata : - https://www.ceicdata.com/datapage/en/thailand/flight-and-passenger-statistics/international-flight-passenger-volume-arrival

- Meteostat weather API : https://api.meteostat.net

- Cost of living index : https://www.numbeo.com/cost-of-living/rankings

- Population Index : http://worldpopulationreview.com

- 


## The Goal
- The goal of the project is to predict whether the current boost in tourism in certain parts of the world are seasonal or just random or these rising popularity is here to stay in the long run. A few years ago the situation got to a point where schools had to be converted into hotels to accomodate the high number of tourists in Iceland.


- Forecasting trends or seasonality in the tourism industry using different timeseries models.

- Passengers flight data of both domestic and international flights

- By using different timeseries models such as ARMA, ARIMA, SARIMA, SARIMAX etc, checking seasonality and trends the model will be able to predict whether the current boom in popularity of certain tourist destinations are in the longrun beneficial for longterm business investments etc.

- I am looking foward to apply what i have learnt in the past few weeks to make the best most of my final project.


# EDA 

Each city showed very similar shape and seasonal trends: the differences are between the magnitude and intercept.
<p align="center">
  <img width="700" height="600" src="https://github.com/jankomah/Travel_Trends_Project/blob/master/images/world_graph.jpeg">
</p>
Upon looking at the data on an average basis across all cities, we can see a clear upward trend and seasonality, with winter being an unpopular time to travel, and summer a popular time. Travellers tend to travel more around the world in large numbers during the summer peroids to different locations around the world.
<p align="center">
  <img width="800" height="600" src="https://github.com/jankomah/Travel_Trends_Project/blob/master/images/total_activity.jpeg">
</p>
With a closer look at monthly volume, summer shows the greatest amount of travel, especially July and August, and winter the least, despite an expectation that holidays might increase travel in December.
<p align="center">
  <img width="800" height="600" src="https://github.com/jankomah/Travel_Trends_Project/blob/master/images/monthly_boxplot.jpeg">
</p>
My main obsession was in looking at which destinations had the largest overall increases in visitors from 2010-2018.
<p align="center">
  <img width="900" height="500" src="https://github.com/jankomah/Travel_Trends_Project/blob/master/images/Screenshot%202020-03-02%20at%2023.12.47.png">
</p>



















































