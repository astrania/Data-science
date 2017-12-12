/*
SUMMARY
Our team performed a thorough review of the proposal and the weekly sales data for
Phoenix and Tucson, successfully accomplishing the scope of services requested in the
request for proposal.

RESULTS
PHOENIX
A Single ESM was developed for the weekly forecasts for Phoenix. The model was then
tested on the validation dataset resulting in a mean absolute percent error (MAPE) of
1.489%.
TUCSON
A Brown/Double ESM was developed for the weekly forecasts for Tucson. Strategies to
account for this deterministic trend component are outlined below. The model was
tested on the validation dataset resulting in a MAPE of 5.191%.
DATA STATIONARITY
An Augmented Dickey-Fuller test was performed on both the Phoenix and Tucson Sales
data. The Phoenix data was found to lack stationary around a single mean, but was
found to be stationary about zero after taking the lag one difference. The Tucson data
was confirmed to be stationary around a deterministic trend .
RECOMMENDATIONS
? The marketing analysis team at Modern Retail, Inc. should account for the lack of
stationarity in the Phoenix dataset by taking the first lag difference of the data
before attempting further time-series modeling.
? The marketing analysis team at Modern Retail, Inc. should account for the
deterministic trend found in the Tucson dataset. To do so, our team suggests
running a regression of the data against time to estimate the trend.

METHODOLOGY
To begin the process, our team thoroughly reviewed the data for the weekly sales of
Phoenix and Tucson. The data was split in Excel into a training and validation data set.
The last 16 weeks of data were set aside for the validation data set.
Next, we created a weekly forecast for sales in both Phoenix and Tucson. This was done
using different ESMs on the training dataset. Note that the models for each store were
built separately.
After that, evaluation of the data from each store was performed in order to allow for
further time series modeling. The stationarity of the sales for each store was checked.
Our team used the Augmented Dickey-Fuller tests up to lag 2 for this, as recommended
by Modern Retail’s analysts. Based on the results of this, strategies were formulated for the
marketing analysis team to make the data stationary.
Finally, accuracy was tested for the forecasts we produced. Our team felt that using the
mean absolute percentage error (MAPE) was the best measurement to use since
Modern Retail, Inc. is familiar and comfortable using it. The resulting statistics are further
outlined in the analysis section below.

ANALYSIS
To begin the analysis, a time series procedure was run in SAS to determine if there were
any trends to the data and to confirm that seasonality was not present, as Modern
Retail’s analysts reported. Each location was studied using weeks as the time interval.
PHOENIX

MODELING
It was discovered that the Phoenix location did not demonstrate a trend or seasonal
component. Therefore, a Single ESM was chosen to model the data from the Phoenix
location (Figure 1).

ACCURACY
To test the model created by the training data, the predictions from the Single ESM were
merged with the validation dataset that was set aside at the beginning of the process.
The absolute percent error (APE) was calculated, and using the means procedure in SAS,
a MAPE value was reported as 1.49%.

STATIONARITY
From an Augmented Dickey-Fuller Test up to a lag of 2, we found a lack of stationarity in
the data for the Phoenix location. Before building further time series models, we
recommend taking the first difference of the data. A further Augmented Dickey-Fuller
Test showed that taking the first difference does create stationarity in the data.

TUCSON
MODELING
When reviewing the Tucson data, it was discovered that a trend component was
present. Therefore, our analysts chose to run the following three models and see which
one performed the best: Brown/Double, Holt/Linear, and Damped Trend. The
Brown/Double model was selected as the model of best fit (Figure 2), as explained
below.

ACCURACY
To test the models created by the training data, the predictions from the three models
specified above were merged with the validation dataset. The APE for each was
calculated, and using the means procedure in SAS, MAPEs were also calculated. After
comparing the MAPEs for the three models, the Brown/Double model had the lowest
MAPE reported as 5.1916% and therefore was chosen to move forward in the process.

STATIONARITY
From an Augmented Dickey-Fuller Test up to a lag of 2, it was determined that the data
was stationary around a deterministic trend for the Tucson location. Future modeling
should regress this data on time to remove the trend component.

CONCLUSION
Modern Retail, Inc. should use the single ESM for Phoenix and the Brown/Double ESM for
Tucson in order to accurately forecast sales in these locations. If you choose to hire our
team for further consultation, we will provide solutions to account for lack of stationarity
in the data to enable future modeling capabilities.*/

libname tseries '\Google Drive\Mystery_Machine\Time Series\HW2';

*Import test data csv and convert to SAS file;
data tseries.testdata;
	infile "\Documents\Homwork Data\Time\HW2\testdata.csv" dlm=',';
	input Date :mmddyy10. SALES_PH SALES_TU;
	label SALES_PH='Phoenix Sales' SALES_TU='Sales Tucson';
	format date date9.;
run;

proc print data=tseries.testdata;
run;

*Import validation data csv and convert to SAS file;
data tseries.validation;
	infile "\Documents\Homwork Data\Time\HW2\validation.csv" dlm=',';
	input Date :mmddyy10. SALES_PH SALES_TU;
	label SALES_PH='Phoenix Sales' SALES_TU='Sales Tucson';
	format date date9.;
run;

proc print data=tseries.validation;
run;

*Create week variable for time interval;
data tseries.testdata;
	set tseries.testdata;
	Week=week(date);
run;

*Decomposition of test data;
proc timeseries data=tseries.testdata plots=(series decomp);
	id date interval=week;
	var SALES_PH SALES_TU;
run;
*No trend or seasonal component for Phoenix, we will use a single ESM;
*Found trend component for Tucson, we will use an additive Linear/Holt ESM;

*Create single ESM for Phoenix;
*Chose lead of 16 because validation set has 16 observations;
proc esm data=tseries.testdata print=all plot=all lead=16 outfor = single;
	id date interval=week;
	forecast SALES_PH / model=simple;
run;

*Create 3 trend ESMs for Tucson: Double/Brown, Linear/Holt, and Damp;
proc esm data=tseries.testdata print=all plot=all lead=16 outfor = double;
	forecast SALES_TU / model=double;
run;


proc esm data=tseries.testdata print=all plot=all lead=16 outfor = linear;
	forecast SALES_TU  / model=linear;
run;


proc esm data=tseries.testdata print=all plot=all lead=16 outfor = damp;
	forecast SALES_TU  / model=damptrend;
run;

*Merge last 16 observations (predictions) with validation data, calculate APE for PHOENIX;
data mergesingle; 
	merge tseries.validation
		Single(firstobs = 245); 
	keep date SALES_PH predict lower upper APE;
	APE = abs(SALES_PH - predict)/SALES_PH *100; 
run;
*Calculate MAPE from APE for PHOENIX;
proc means data = mergesingle;
	var APE;
run;
*MAPE = 1.4899506;

*TUCSON DOUBLE;
*Merge last 16 observations (predictions) with validation data, calculate APE for Tucson;
data mergedouble; 
	merge tseries.validation
		Double(firstobs = 245); 
	keep date SALES_TU predict lower upper APE;
	APE = abs(SALES_TU - predict)/SALES_TU *100; 
run;

*Calculate MAPE from APE for Tucson;
proc means data = mergedouble;
	var APE;
run;
*MAPE DOUBLE 5.1916504;
*Double is the lowest MAPE of the three trend models, we will use this model for Tucson;

*TUCSON LINEAR;
*Merge last 16 observations (predictions) with validation data, calculate APE for Tucson;
data mergelinear; 
	merge tseries.validation
		Linear(firstobs = 245); 
	keep date SALES_TU predict lower upper APE;
	APE = abs(SALES_TU - predict)/SALES_TU *100; 
run;

*Calculate MAPE from APE for Tucson;
proc means data = mergelinear;
	var APE;
run;
*MAPE LINEAR 6.1556557;

*TUCSON DAMP;
*Merge last 16 observations (predictions) with validation data, calculate APE for Tucson;
data mergedamp; 
	merge tseries.validation
		Damp(firstobs = 245); 
	keep date SALES_TU predict lower upper APE;
	APE = abs(SALES_TU - predict)/SALES_TU *100; 
run;

*Calculate MAPE from APE for Tucson;
proc means data = mergedamp;
	var APE;
run;
*MAPE DAMP 6.9521024;

*Dicky Fuller Test for Phoenix;
proc arima data=tseries.testdata plot=all;
	identify var=SALES_PH nlag=16 stationarity=(adf=2);
	identify var=SALES_PH(1) nlag=16 stationarity=(adf=2);
run;
quit;
*We found significance in single mean dicky fuller test, so data is stationary around the mean;

*Dicky Fuller Test for Tucson;
proc arima data=tseries.testdata plot=all;
	identify var=SALES_TU nlag=16 stationarity=(adf=2);
run;
quit;
*We found significance in trend mean dicky fuller test, so data is stationary around the trend- there is a deterministic trend;
