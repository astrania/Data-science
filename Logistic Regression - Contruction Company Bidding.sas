/* Logistic Regression Final Project:
Our consulting group has been tasked with helping AA Construction to make
more informed decisions on bidding for future projects. In order to help AA
Construction save both time and money, our team has devised a more efficient
process to predict the probability AA Construction will win a bid on a project. In
the following report, we have outlined the model we have built and further
information AA Construction can utilize in order to make more informed decisions
on whether or not to invest resources on future projects.

Results
The logistic model predicting the probability AA Construction wins a bid using the
following predictors was built and validated:
Construction Sector, Region of Country, Competitors B, E, F, G,
H, J, Estimated Cost, Difference in Bid and Estimated Cost,
Estimated Construction Time, and Number of Competitor Bids

Our model’s Brier score is 0.099 and its corresponding c-statistic is 0.89 on out-ofsample
data. Figure 1 below shows the Receiver Operating Characteristic (ROC)
curve for the revised model.

Recommendations
? Bidding higher over the expected cost of the project significantly decreases the
chances AA Construction will win a bid. All else equal, a one-million-dollar increase
in bid price results in 0.661 times the odds of winning a bid.
? AA construction has higher odds of winning bids in certain construction sectors. The
highest odds are associated with bidding in the power industry.
? The logistic model can only be used after predicted costs are estimated. The odds
ratio estimates inform which projects to bid on, and the model can then be used
to help refine bid amounts to maximize the likelihood of winning the bid.

Methodology
PRELIMINARY ANALYSIS
We split the dataset containing bid history over the past three years into training and
validation data sets. Following that, we began our process of investigating the variables
in the data for model building. First, we checked for association among variables using
frequency tables. Then, the distributions of predictor variables were reviewed in a
univariate procedure.
Variables were examined for collinearity and adjusted appropriately. Linearity was
checked using the Box-Tidwell transformation on continuous predictors. Finally, interactions
were investigated based on those variables that appeared to have the highest odds ratio
estimates.

MODEL REFINEMENT
To further refine the model, backward selection was used to trim insignificant variables. An
alpha of 0.05 was chosen, as it a commonly accepted significance level for this test. Based
on the results, certain variables found to be insignificant to the model were removed.
Following that, we built classification tables. Youden's statistic was used to calculate the
classification cutoff probability that minimized both false-positives and false-negatives.
The ability of our model to classify was tested by scoring validation data. The calibration
of the predicted probabilities of our data was also tested. Odds ratios were interpreted
from coefficients estimated using all the available data.

Analysis
DATASET MANAGEMENT
The data was split as follows: 80% was used for training the logistic model and 20% of the
data was set aside for validation. Stratification was used to ensure every category was
represented in both the training and validation data. Winning bid price was not used in
models, as that information is not attainable prior to a bid being chosen.

VARIABLE INVESTIGATION
Categorical variables were investigated for association with winning bids and checked for
separation by creating cross tabulation tables. The distributions of the continuous
predictors were plotted as histograms.

COLLINEARITY
All variables were fit to a logistic regression model. The predicted probabilities for the
observations in that model were used as weights in a linear regression using those same
predictors. The variance inflation factors (VIF) in that linear regression were checked for
evidence of collinearity.
Bid Price and Estimated Cost had VIF values over 400, far over the chosen cutoff of 10. A
new variable was created, Bid Difference, calculated as the difference between the bid
price and the estimated cost. Bid Difference, as well as the Estimated Cost, were
maintained as predictors in the model; while Bid Price was dropped. Collinearity was
rechecked, and no VIF values were above 10.

LINEARITY
The Box-Tidwell transformation was used to test if the predictors were linearly related to the
link function of the model. Bid difference was found to contribute a significant amount of
information to the model with its higher-order terms. Bid Difference, however, is a variable
the team wanted to interpret for use in informing AA Construction. The nonlinearity of Bid
Difference was therefore ignored in this model.

INTERACTION
Interaction was checked between Estimated Years to Complete and Number of
Competitor Bids because their odds ratio estimates deviated furthest from one in our initial
models compared to other continuous predictors. The interaction term was found to add
significant information to the model, assessed with the AIC statistic. This term was
maintained in the model.

INFLUENTIAL OBSERVATIONS
Plots of influential observation statistics were created for the model. Some observations
show high leverage, influencing the parameter estimates in the model. None of these
leverages were deemed egregious enough to be removed from the model or investigated
further.

MODEL REFINEMENT
The profile likelihood estimates for odds ratio for some of the predictors in the model had
confidence intervals containing one. In an attempt to remove these predictors, backward
selection was used, rejecting variables at a significance level of 0.05. The effects of bids
from competitors A, C, D, and I were removed from the model.

FINAL MODEL
The validation data was scored using the final model. Classification tables were built using
the training data. The cost of falsely predicting winning a bid and falsely failing to predict
were assumed to be equal. The focus of the model building here was to produce
probabilities and interpret odds ratio estimates; not classify observations that have already
occurred. Using Youden’s index, an optimal cutoff probability was chosen for
classification. Figure 3 below shows the classification of observations from the validation
data.

Figure 3- Classification Table
Predicted /
Actual
Win Bid Did Not Win Bid Total
Win Bid 11 ~ 10.5% 6 ~ 5.7% 17
Did Not Win Bid 8 ~ 7.6% 80 ~ 76.2% 88
Total 19 86 105
In the table, you can see that 57.9% of the bids predicted to win actually won. The
calibration and ROC curves for the model on the validation data are presented in Figures
1 on page 2, the results section of this report.

The model was fit using the entire data set. Profile likelihood estimates for the odds ratios
of the different predictors are included in Figure 4 below, excluding Estimated construction
time and number of competitor bids, as those variables were modeled with interaction.

Increasing the bid price one million dollars over the estimated cost, holding all else
equal, is associated with approximately 0.661 times the likelihood of winning the
bid. Estimated costs increasing by one million dollars, holding all else equal, is
associated with approximately 1.037 times the likelihood of winning the bid

Conclusion
The model built should be used to assess how likely a project bid is to be won. That
predicted probability can be used to adjust a bid price to maximize win likelihood. The
odds ratio estimates from the model can also be used to target high-likelihood projects.
For instance, a power sector project set in the western region of the country has higher
odds of winning the bid than a military project set in the southwest.
*/

libname logistic "\Mystery_Machine\Logistic\Project";

*Macro for categorical predictors - for simple class statements in initial investigations;
%let cat =  Sector Region_of_Country Competitor_A Competitor_B Competitor_C Competitor_D Competitor_E Competitor_F Competitor_G Competitor_H Competitor_I Competitor_J;



/* INVESTIGATION */

*Print Head of Construction Data;
proc print data = logistic.construction(obs = 20);
run;

*Look at Dataset contents and missing values;
proc contents data = logistic.construction;
run;
proc freq data = logistic.construction nlevels;
	tables _all_ / noprint;
run;



/* VALIDATION SEPARATION */

*Sort data to prepare for validation selection;
proc sort data = logistic.construction
		out = const_sort;
		by &cat;
run;

*Select validation data;
proc surveyselect data=const_sort method=srs rate=0.2 seed = 1234
				  out=const_sort_sel outall;
				  strata &cat; *Ensure every category is represented in both datasets; 
run;
data logistic.construction_train logistic.construction_valid;
	set const_sort_sel;
	if Selected = 1 then output logistic.construction_train;
	else output logistic.construction_valid;
	drop selected selectionprob samplingweight;
run;



/* VARIABLE INVESTIGATION FOR MODEL BUILDING*/

*Frequency Tables for Association investigation;
proc freq data = logistic.construction_train nlevels;
	tables Win_Bid * (sector region_of_country
		competitor_A competitor_B
		competitor_C competitor_D
		competitor_E competitor_F
		competitor_G competitor_H
		competitor_I competitor_J) / chisq;
run;

*Distributions of Continuous Predictors;
proc univariate data = logistic.construction_train noprint;
	histogram Bid_Price__Millions_ 
		Estimated_Cost__Millions_ 
		Estimated_Years_to_Complete 
		Number_of_Competitor_Bids;
run;



/* INITIAL MODEL AND ASSUMPTIONS */
*Initial Model Includes All Reasonable Predictors, will reduce with selection;

*Check for Collinearity;
*Output predicted probabilities;
proc logistic data = logistic.construction_train;
	class &cat;
	model Win_Bid(Event = 'Yes') = &cat Estimated_Years_To_Complete Bid_Price__Millions_ Estimated_Cost__Millions_ Number_of_Competitor_Bids ;
	output out = train_p PRED = phat;
run;
*Create weight for each observation from predicted probabilities;
Data train_pred;
	set train_p;
	Win_Bid_Num = 0;
	if Win_Bid = 'Yes' then Win_Bid_Num = 1;
	w = phat*(1-phat);
run;
*Look at VIF for variables in weighted regression model;
proc reg data = train_pred;
	weight w;
	model Win_Bid_Num = Competitor_A Competitor_B Competitor_C Competitor_D
						Competitor_E Competitor_F Competitor_G Competitor_H
						Competitor_I Competitor_J
						Estimated_Years_To_Complete Bid_Price__Millions_ Estimated_Cost__Millions_ Number_of_Competitor_Bids 
						/ TOL VIF;
run;
quit;
*Bid Price and Estimated Cost Very Collinear - Transform in Datasets;

*Difference two variables - measure of "Overbidding";
data logistic.construction_train;
	set logistic.construction_train;
	Bid_Difference = (Bid_Price__Millions_ - Estimated_Cost__Millions_);
run;
data logistic.construction_valid;
	set logistic.construction_valid;
	Bid_Difference = (Bid_Price__Millions_ - Estimated_Cost__Millions_);
run;
proc univariate data = logistic.construction_train noprint;
	histogram Bid_Difference;
run;
proc sql;
	select mean(Bid_Difference) as Mean_Diff, Win_Bid
	from logistic.construction_train
	group by Win_bid;
quit;

/* Re-check collinearity after transforming collinear variables*/
*Same process as before, but with new variable;
proc logistic data = logistic.construction_train;
	class &cat;
	model Win_Bid(Event = 'Yes') = &cat Estimated_Years_To_Complete Bid_Difference Estimated_Cost__Millions_ Number_of_Competitor_Bids ;
	output out = train_p PRED = phat;
run;
Data train_pred;
	set train_p;
	Win_Bid_Num = 0;
	if Win_Bid = 'Yes' then Win_Bid_Num = 1;
	w = phat*(1-phat);
run;
proc reg data = train_pred;
	weight w;
	model Win_Bid_Num = Competitor_A Competitor_B Competitor_C Competitor_D
						Competitor_E Competitor_F Competitor_G Competitor_H
						Competitor_I Competitor_J
						Estimated_Years_To_Complete Bid_Difference Estimated_Cost__Millions_ Number_of_Competitor_Bids 
						/ TOL VIF;
run;
quit;
/* No significant collinearity */

/* Checking for linearity */
*Box-Tidwell transform on continuous predictors;
data logistic.construction_train;
	set logistic.construction_train;
	Estimated_Years_To_CompleteL = Estimated_Years_To_Complete*log(Estimated_Years_To_Complete);
	Bid_DifferenceL = Bid_Difference*log(Bid_Difference);
	Number_of_Competitor_BidsL = Number_of_Competitor_Bids*log(Number_of_Competitor_Bids);
	Estimated_Cost__Millions_L = Estimated_Cost__Millions_*log(Estimated_Cost__Millions_);
run;

proc logistic data = logistic.construction_train;
	class &cat;
	model Win_Bid(Event = 'Yes') = &cat 
		Estimated_Cost__Millions_ Estimated_Years_To_Complete Bid_Difference Number_of_Competitor_Bids;
run; /* AIC = 137 Bid-Difference est. -0.4982*/

proc logistic data = logistic.construction_train;
	class &cat;
	model Win_Bid(Event = 'Yes') = &cat 
		Estimated_Cost__Millions_ Estimated_Years_To_Complete Bid_Difference Number_of_Competitor_Bids 
		Estimated_Cost__Millions_L Estimated_Years_To_CompleteL Bid_DifferenceL Number_of_Competitor_BidsL;
run; /* AIC = 122 Bid-DifferenceL est. 1.0296*/

/* Transform Bid_Difference */

data logistic.construction_train;
	set logistic.construction_train;
	Bid_Dif_Trans = Bid_Difference**(-1.0296/0.4982 + 1);
run;
data logistic.construction_valid;
	set logistic.construction_valid;
	Bid_Dif_Trans = Bid_Difference**(-1.0296/0.4982 + 1);
run;
proc univariate data = logistic.construction_train noprint;
	histogram Bid_Dif_Trans;
run;
proc sql;
	select mean(Bid_Dif_Trans) as Mean_Diff, Win_Bid
	from logistic.construction_train
	group by Win_bid;
quit;

/* Ultimately, transform of Bid Price was not used to maintain interpretability */

/* Checking for Interaction */

proc logistic data = logistic.construction_train;
	class &cat;
	model Win_Bid(Event = 'Yes') = Sector Region_of_Country Competitor_A Competitor_B Competitor_C Competitor_D Competitor_E Competitor_F Competitor_G Competitor_H Competitor_I Competitor_J 
								Estimated_Cost__Millions_ Bid_Difference Estimated_Years_To_Complete|Number_of_Competitor_Bids @2;
run;

proc logistic data = logistic.construction_train;
	class &cat;
	model Win_Bid(Event = 'Yes') = Sector Region_of_Country Competitor_A Competitor_B Competitor_C Competitor_D Competitor_E Competitor_F Competitor_G Competitor_H Competitor_I Competitor_J 
								Estimated_Cost__Millions_ Bid_Difference Estimated_Years_To_Complete Number_of_Competitor_Bids;
run; /* Keeping EST-NUMBIDS interaction*/

/* Refining Model for significance, classification, and calibration */

/* Backward Selection to trim insignificant variables */
proc logistic data = logistic.construction_train;
	class &cat;
	model Win_Bid(Event = 'Yes') = Sector Region_of_Country Competitor_A Competitor_B Competitor_C Competitor_D Competitor_E Competitor_F Competitor_G Competitor_H Competitor_I Competitor_J 
								Bid_Difference Estimated_Cost__Millions_ Estimated_Years_To_Complete|Number_of_Competitor_Bids 
		/ selection = backward slstay = 0.05;
run;

*Build classification tables;
proc logistic data = logistic.construction_train;
	class Sector Region_of_Country Competitor_B Competitor_E Competitor_F Competitor_G Competitor_H Competitor_J / param = ref;
	model Win_Bid(Event = 'Yes') = Sector Region_of_Country Competitor_B Competitor_E Competitor_F Competitor_G Competitor_H Competitor_J
									Bid_Difference Estimated_Cost__Millions_ Estimated_Years_To_Complete|Number_of_Competitor_Bids
								/ ctable pprob= 0 to 0.98 by 0.02;
	ods output classification = classtable;
run;

/* Youden Statistic */
data classtable;
	set classtable;
	youden = sensitivity + specificity - 100;
run;
proc sort data=classtable;
	by descending youden;
run;
proc print data=classtable;
run;
/* 0.18 cutoff maximizes Youden statistic*/

/* Checking Calibration */

*output predicted probabilities for training and validation data;
proc logistic data=logistic.construction_train;
	class Sector Region_of_Country Competitor_B Competitor_E Competitor_F Competitor_G Competitor_H Competitor_J / param = ref;
	model Win_Bid(Event = 'Yes') = Sector Region_of_Country Competitor_B Competitor_E Competitor_F Competitor_G Competitor_H Competitor_J
									Bid_Difference Estimated_Cost__Millions_ Estimated_Years_To_Complete|Number_of_Competitor_Bids;
	output out = construction_train_scored p = predicted;
	score data=logistic.construction_valid out=construction_valid_scored outroc=construction_valid_scoredroc fitstat; 
run;

*Edit outputs from regression, also clasify validation based on Youden optimized cutoff;
data construction_valid_scored;
	set construction_valid_scored;
	if P_Yes > 0.18 then Predicted = 1; else Predicted = 0;
	Win_Bid_Num = 0;
	if Win_Bid = 'Yes' then Win_Bid_Num = 1;
run;
data construction_train_scored;
	set construction_train_scored;
	Win_Bid_Num = 0;
	if Win_Bid = 'Yes' then Win_Bid_Num = 1;
run;

*Classification table using cutoff;
proc freq data=construction_valid_scored;
	tables Win_Bid*Predicted / nocol nopercent norow;
run;
quit;

*Calibration for training;
proc sgplot data=construction_train_scored;
	scatter x=predicted y=Win_Bid_Num;
/* overlay loess smooth with conf limits */
	loess x=predicted y=Win_Bid_Num / clm;
/* add diagonal line */
	lineparm x=0 y=0 slope=1;
run; 

*Calibration for validation, overconfidence evident;
proc sgplot data=construction_valid_scored;
	scatter x=P_Yes y=Win_Bid_Num;
/* overlay loess smooth with conf limits */
	loess x=P_Yes y=Win_Bid_Num / clm;
/* add diagonal line */
	lineparm x=0 y=0 slope=1;
run;
