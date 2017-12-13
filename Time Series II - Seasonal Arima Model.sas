/*
SUMMARY
Our analysts developed a seasonal ARIMA model that improves upon the Exponential Smoothing
model (ESM) previously used to forecast hourly temperatures at the main retail location in Phoenix,
AZ. When tested on the validation data, the ARIMA model had a MAPE score of 1.53%, lower than the
ESM’s MAPE of 1.98%. The ARIMA model’s low forecasting error makes it a strong candidate for use in
future analysis, such as the proposed exploration of the association between extreme outdoor
temperatures and sales.

RESULTS
We determined that the best ARIMA model for Phoenix temperature data is an ARIMA(2,0,1) model
that uses dummy variables to account for seasonality. As in our previous report, we confirmed
stationarity before modeling. In this case, differencing was not necessary. Our MAPE was better by
almost half a percentage point (-0.45%) compared to the ESM model with a lag difference of 1.  
We forecasted hourly temperatures for the first day of September using the ARIMA(2,0,1) model with
seasonal dummy variables. Figure 1 below displays August temperature data as well as predicted
temperatures for September with a 95% confidence band. The exact predicted values and their
confidence intervals are available in the appendix.

RECOMMENDATIONS
We recommend using the ARIMA(2,0,1) model with seasonal dummy variables for future analyses. This
model generates more accurate predictions than the ESM model, as indicated by a lower MAPE.

METHODOLOGY
SEASONALITY AND TREND
We began our analysis by decomposing the data to investigate possible seasonality and trend. As seen
in the second panel of Figure 2 below, there is a clear seasonal pattern that repeats each 24-hour
period, but no evidence of consistent trend over the course of a month. We verified the deterministic
nature of the seasonal component using a Seasonal Augmented Dicky-Fuller (ADF) test.
We accounted for seasonality in our model by creating dummy variables for each hour of the 24-hour
season. Although this method does not assign greater weights to more recent observations, the low
MAPE score from the resulting model indicates that variable weights were not needed to generate
accurate predictions.

Lastly, we examined the decomposition plot to check for additive or multiplicative structure in the
data. Seeing a relatively constant pattern (as opposed to the funnel shape characteristic of
multiplicative series), we decided to proceed with an additive model.

DATA STATIONARITY
Before proceeding to modeling, we confirmed stationarity of the data using a single mean ADF test.
The results of the ADF test indicated that differencing was not required, ie that the data was stationary
around the season.

MODEL SELECTION
As requested, our team built multiple models - one that utilized both stochastic seasonality, and one
that had deterministic seasonality.  For stochastic seasonality, we built a model with the 24th
difference of the temperature data.  For deterministic seasonality, we investigated two ways to deal
with the clear seasonality of the data: seasonal dummies, and incorporating a cosine.  Of these two
methods seasonal dummies best handled the seasonality of the data, as indicated by the white noise
plot.  Overall, the deterministic seasonal dummy model outperformed the stochastic differenced
model, as shown by a MAPE of 2.35% for the stochastic model compared to a MAPE of 2.14% for the
deterministic model.

This improvement in MAPE led us to choose seasonal dummy variables as the solution to deal with the
seasonality of the temperature data.  However, the white noise of the model was still not ideal.
 Further model analysis showed that an Autoregressive(2), Moving Average(1) model was the best for
this data. The significantly improved white noise can be seen in Figure 3.  The overall MAPE for this
model is 1.53%, a significant improvement from Modern Retail’s previous ESM models.

CONCLUSION
To forecast temperatures for Phoenix, AZ, we suggest implementing the ARIMA(2,0,1) model with
seasonal dummy variables. This model has a lower MAPE, 1.53%, than the previous best ESM model
utilized by Modern Retail, Inc, at 1.98%.
*/
libname hw4 "\Time Series 2\Homework\HW2";

/*Read training set into sas*/
Data hw4.aug_train;
	infile "\Time Series 2\Homework\HW2\AUGUST_TRAIN_CLEAN.csv" firstobs=2 DSD MISSOVER;
	input Date Time DryBulb hour;
run;

/*Read valid set into sas*/
Data hw4.sept_valid;
	infile "\Time Series 2\HW 4\SEPTEMBER_VALID_CLEAN.csv" firstobs=2 DSD MISSOVER;
	input Date Time DryBulb;
run;

/*Visualize Data*/
proc arima data=hw4.aug_train plot=all;
	identify var=DryBulb;
quit;

/*Run adf test*/
proc arima data=hw4.aug_train plot=all;
	identify var=DryBulb nlag=60 stationarity=(adf=5);
run;
quit;


proc arima data=hw4.aug_train plot=all;
	identify var=DryBulb nlag=60 stationarity=(adf=5);
	*identify var=Temperature nlag=60 stationarity=(adf=2 dlag=12);
	*identify var=Temperature(12) stationarity=(adf=2);
	*estimate p=(1) q=(1,12);
run;
quit;

/*Create hourly dummy variables*/
data hw4.aug_train_2;
	set hw4.aug_train;
	if hour=1 then hour1 = 1; else hour1=0;
	if hour=2 then hour2 = 1; else hour2=0;
	if hour=3 then hour3 = 1; else hour3=0;
	if hour=4 then hour4 = 1; else hour4=0;
	if hour=5 then hour5 = 1; else hour5=0;
	if hour=6 then hour6 = 1; else hour6=0;
	if hour=7 then hour7 = 1; else hour7=0;
	if hour=8 then hour8 = 1; else hour8=0;
	if hour=9 then hour9 = 1; else hour9=0;
	if hour=10 then hour10 = 1; else hour10=0;
	if hour=11 then hour11 = 1; else hour11=0;
	if hour=12 then hour12 = 1; else hour12=0;
	if hour=13 then hour13 = 1; else hour13=0;
	if hour=14 then hour14 = 1; else hour14=0;
	if hour=15 then hour15 = 1; else hour15=0;
	if hour=16 then hour16 = 1; else hour16=0;
	if hour=17 then hour17 = 1; else hour17=0;
	if hour=18 then hour18 = 1; else hour18=0;
	if hour=19 then hour19 = 1; else hour19=0;
	if hour=20 then hour20 = 1; else hour20=0;
	if hour=21 then hour21 = 1; else hour21=0;
	if hour=22 then hour22 = 1; else hour22=0;
	if hour=23 then hour23 = 1; else hour23=0;
	if hour=24 then hour24 = 1; else hour24=0;
run;

/*---------------------------------------------------------------------------------*/
/*Create data for forecasting, merge into aug_merged*/

data hw4.blank;
	input Date Time DryBulb hour1 hour2 hour3 hour4 hour5 hour6 hour7 hour8
			hour9 hour10 hour11 hour12 hour13 hour14 hour15 hour16 hour17 hour18 hour19
			hour20 hour21 hour22 hour23 hour24;
datalines;
	20160901	.	. 	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1	0
	20160901	.	.	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	1
;

data hw4.aug_merged;
	set hw4.aug_train_2 hw4.blank;
run;

/*---------------------------------------------------------------------------------*/

/*Run proc arima with the dummy variables with difference, forecast for Sept 1*/
proc arima data=hw4.aug_merged plot=forecasts(all);
	identify var=DryBulb(1) nlag=30 crosscorr=(hour1 hour2 hour3 hour4 hour5 hour6 hour7 hour8
			hour9 hour10 hour11 hour12 hour13 hour14 hour15 hour16 hour17 hour18 hour19
			hour20 hour21 hour22 hour23 hour24 ) stationarity=(adf=5);
	estimate input=(hour1 hour2 hour3 hour4 hour5 hour6 hour7 hour8
			hour9 hour10 hour11 hour12 hour13 hour14 hour15 hour16 hour17 hour18 hour19
			hour20 hour21 hour22 hour23 hour24 );
	forecast lead=24;
run;
quit;


/*---------------------------------------------------------------------------------*/
/*Stochastic Model looking at the 24th difference*/
proc arima data=hw4.aug_merged plot=all;
	identify var=DryBulb(24) nlag=30 stationarity=(adf=5);
	forecast lead=24;
run;
quit;

/*Stochastic model with 24th difference with ARIMA(2,1)*/
proc arima data=hw4.aug_merged plot=all;
	identify var=DryBulb(24) nlag=30 stationarity=(adf=5);
	estimate p=2 q=1;
	forecast lead=24;
run;
quit;

/*---------------------------------------------------------------------------------*/
/*ARIMA Model - DUMMY VARIABLE DATA SET*/
proc arima data=hw4.aug_merged plot=forecasts(all);
	identify var=DryBulb nlag=30 crosscorr=(hour1 hour2 hour3 hour4 hour5 hour6 hour7 hour8
			hour9 hour10 hour11 hour12 hour13 hour14 hour15 hour16 hour17 hour18 hour19
			hour20 hour21 hour22 hour23 hour24 ) stationarity=(adf=5);
	estimate p=2 q=1 input=(hour1 hour2 hour3 hour4 hour5 hour6 hour7 hour8
			hour9 hour10 hour11 hour12 hour13 hour14 hour15 hour16 hour17 hour18 hour19
			hour20 hour21 hour22 hour23 hour24 );
	forecast lead=24;
run;
quit;

/*Box Jenkins Seasonal ARIMA Model - ORIGINAL DATA SET*/
proc arima data=hw4.aug_train plot=forecasts(all);
	identify var=DryBulb nlag=30 stationarity=(adf=5);
	estimate p=2 q=1;
	forecast lead=24;
run;
quit;

/*---------------------------------------------------------------------------------*/

/*Trying Cosine Model - Trig Model*/
data hw4.aug_train_trig;
	set hw4.aug_train;
	pi = constant('PI');
	x1=cos(2*pi*1*_n_/12); x2=sin(2*pi*1*_n_/12);
	x3=cos(2*pi*2*_n_/12); x4=sin(2*pi*2*_n_/12);
	x5=cos(2*pi*3*_n_/12); x6=sin(2*pi*3*_n_/12);
	x7=cos(2*pi*4*_n_/12); x8=sin(2*pi*4*_n_/12);
	x9=cos(2*pi*5*_n_/12); x10=sin(2*pi*5*_n_/12);
run;


proc arima data=hw4.aug_train_trig plot=forecasts(all);
	identify var=DryBulb nlag=30 crosscorr=(x1 x2 x3 x4 x5 x6 x7 x8 x9 x10);
	estimate input=(x1 x2 x3 x4 x5 x6 x7 x8 x9 x10);
	*estimate input=(x1 x2 x4 x10);
	*forecast lead=24;
run;
quit;
