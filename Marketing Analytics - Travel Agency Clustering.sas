/*An online travel company is trying to identify consumer segments in the current travel 
market to offer promotional packages to targeted segments. The company provided a sample 
data set of their customers from their customer database about their preferences when 
planning trips/vacation and purchasing online to a team of consultants (your team).
The sample dataset consists of data from over 3,500 individuals. The company’s Marketing
Executives are interested in no more than 7 and no less than 4 segments. The travel 
company is interested in targeting the most relevant segment(s)and positioning their 
product(s) appropriately. The consulting team should submit a report(8 pages maximum 
excluding Appendices) to the company executives summarizing the team’s recommendation 
and analysis. CLUE: YOU WILL HAVE TO REDUCE YOUR VARIABLE SET USING DIMENSION 
REDUCTION TECHNIQUES BEFORE YOU RUN YOUR CLUSTERING ALGORITHM (USE PRINCIPAL COMPONENT ANALYSIS, 
EXPLORATORY FACTOR ANALYSIS, OR VARIABLE CLUSTERING TECHNIQUES)
Note: You have to convince me (the Managementof the online travel company)
of your segmentation solution and of your positioning strategy. 
*/

proc import datafile = '\MSA\Marketing Analytics\Final Project\domestic travel preferences.csv'
 out = work.preferences
 dbms = CSV;
run;

/*Making sure data loaded*/
proc print data=preferences (obs=20);
run;

/*Looking at the distribution and number of missing*/
proc univariate data=preferences noprint;
histogram Gender Age Trips Q1_1 q1_2 q1_3 q1_4 q1_5 q1_6 q1_7 q1_8 q48_1 q48_2 q48_3 q48_4 q48_5 q48_6 
			q48_7 q48_8 q48_9 q48_10 q48_11 q48_12 q48_13 q48_14 q48_15 q48_16 q48_17 q48_18 q48_19 q48_20
			q48_21 q48_22 booking monthyear purpose decmaker people marital education employment income ethnicity;
inset  nmiss mean median std / position = nw;
run;

/*Dropping all observations with missing rows except for the missing in income, we will bin those*/
%let input = Gender Age Trips Q1_1 q1_2 q1_3 q1_4 q1_5 q1_6 q1_7 q1_8 q48_1 q48_2 q48_3 q48_4 q48_5 q48_6 
				q48_7 q48_8 q48_9 q48_10 q48_11 q48_12 q48_13 q48_14 q48_15 q48_16 q48_17 q48_18 q48_19 q48_20
				q48_21 q48_22 booking monthyear purpose decmaker people marital education employment ethnicity;

data preferences_nonmiss;
	set preferences;
	if cmiss(of &input)  then delete;
run; 

/*Checking for missing again*/
proc means nmiss;
 var Gender Age Trips Q1_1 q1_2 q1_3 q1_4 q1_5 q1_6 q1_7 q1_8 q48_1 q48_2 q48_3 q48_4 q48_5 q48_6 
			q48_7 q48_8 q48_9 q48_10 q48_11 q48_12 q48_13 q48_14 q48_15 q48_16 q48_17 q48_18 q48_19 q48_20
			q48_21 q48_22 booking monthyear purpose decmaker people marital education employment income ethnicity;
run;

proc contents data=preferences_binary;
run;

/*Making Gender, MonthYear, purpose, income  binary*/
data preferences_binary;
	set preferences_nonmiss;
	if Gender = 1 then male = 1; else male =0;

	if monthyear = 2 then Nov = 1; else Nov = 0;
	if monthyear = 3 then Dec = 1; else Dec = 0;
	if monthyear = 4 then Jan = 1; else Jan = 0;
	if monthyear = 5 then Feb = 1; else Feb = 0;
	if monthyear = 6 then Mar = 1; else Mar = 0;
	if monthyear = 7 then Apr = 1; else Apr = 0;
	if monthyear = 8 then May = 1; else May = 0;
	if monthyear = 9 then Jun = 1; else Jun = 0;
	if monthyear = 10 then Jul = 1; else Jul = 0;
	if monthyear = 11 then Aug = 1; else Aug = 0;
	if monthyear = 12 then Sep = 1; else Sep = 0;
	if monthyear = 13 then Oct = 1; else Oct = 0;
	
	if education = 1 then less_hs = 1; else less_hs = 0;
	if education = 2 then completedsome_hs  = 1; else completedsome_hs = 0;
	if education = 3 then highschool = 1; else highschool = 0;
	if education = 4 then undergrad_nondegree = 1; else undergrad_nondegree = 0;
	if education = 5 then undergrad = 1; else undergrad = 0;
	if education = 6 then graduate_nondegree = 1; else graduate_nondegree = 0;
	if education = 7 then graduate = 1; else graduate = 0;

	if purpose = 1 then MajorVacation = 1; else MajorVacation = 0;
    if purpose = 2 then getaway = 1; else getaway  = 0;
    if purpose = 3 then leisure = 1; else leisure = 0;
    if purpose = 5 then business = 1; else business = 0;

    if decmaker = 1 then dec_primary = 1; else dec_primary = 0;
    if decmaker = 2 then dec_other = 1; else dec_other = 0;
    if decmaker = 3 then dec_joint = 1; else dec_joint = 0;

    if marital = 1 then single = 1; else single = 0;
    if marital = 2 then married = 1; else married = 0;
    if marital = 3 then divorced = 1; else divorced = 0;
    if marital = 4 then seperated = 1; else seperated = 0;
    if marital = 5 then widowed= 1; else widowed = 0;
    if marital = 6 then partner= 1; else partner = 0;	

	if employment = 1 then emp_full = 1; else emp_full = 0;
	if employment = 2 then emp_part = 1; else emp_part = 0;
	if employment = 3 then emp_self = 1; else emp_self = 0;
	if employment = 4 then emp_looking = 1; else emp_looking = 0;
	if employment = 5 then emp_notlooking = 1; else emp_notlooking = 0;
	if employment = 6 then emp_retired = 1; else emp_retired = 0;
	if employment = 7 then emp_student = 1; else emp_student = 0;
	if employment = 8 then emp_homemaker = 1; else emp_homemaker = 0;

	if ethnicity = 1 then black = 1; else black = 0;
	if ethnicity = 2 then hispanic = 1; else hispanic = 0;
	if ethnicity = 3 then native = 1; else native = 0;
	if ethnicity = 4 then oriental = 1; else oriental = 0;
	if ethnicity = 5 then white = 1; else white = 0;
	if ethnicity = 6 then mixed = 1; else mixed = 0;
	if ethnicity = 7 then other = 1; else other = 0;
	if ethnicity = 8 then ethnicity_notanswer = 1; else ethnicity_notanswer = 0;

	if income = 1 then income1 = 1; else income1 = 0;
	if income = 2 then income2 = 1; else income2 = 0;
	if income = 3 then income3 = 1; else income3 = 0;
	if income = 4 then income4 = 1; else income4 = 0;
	if income = 5 then income5 = 1; else income5 = 0;
	if income = 6 then income6 = 1; else income6 = 0;
	if income = 7 then income7 = 1; else income7 = 0;
	if income = 8 then income8 = 1; else income8 = 0;
	if income = 9 then income9 = 1; else income9 = 0;
	if income = 10 then income10 = 1; else income10 = 0;
	if income = . then income_miss = 1; else income_miss = 0;
	
	*drop gender monthyear purpose decmaker marital education employment income ethnicity;
run;


/*Standardizing questionnaire variables*/
%let input2 =  Q1_1 q1_2 q1_3 q1_4 q1_5 q1_6 q1_7 q1_8 q48_1 
				q48_2 q48_3 q48_4 q48_5 q48_6 
				q48_7 q48_8 q48_9 q48_10 q48_11 q48_12 q48_13 
				q48_14 q48_15 q48_16 q48_17 q48_18 q48_19 q48_20
				q48_21 q48_22 ;

proc STDIZE	data=preferences_binary out=preferences_stand method=std ;
    var &input2 ;
run;


/*Variable reduction using PCA*/
ods graphics on;
proc princomp data=preferences_stand out=preferences_pca;
	var Q1_1 -- q48_22;
run;
ods graphics off;

/*Variable clustering*/
ods graphics on;
proc varclus data=preferences_stand  ;
	var Q1_1 -- q48_22;
run;
/*Variables to be selected after varclus; q48_6 q48_10 q48_20 q1_5 q48_1 q48_7 q1_6 q48_17 q1_1*/

%let varclus_variables = q48_6 q48_10 q48_20 q1_5 q48_1 q48_7 q1_6 q48_17 q1_1;

/***PCA***/


/*Determining number of clusters*/
ods graphics on;
proc cluster data=preferences_pca method=ward simple noeigen 
				RMSSTD RSQ PSEUDO CCC print=15 outtree=tree;
var prin1 -- prin7 booking;
run;
ods graphics off;
/*4 Clusters*/

/*Dendrogram of cluster solution*/
TITLE2 'Dendrogram of the Cluster Solution';
PROC TREE Data=tree levels=0.000032 OUT=hcluspref ;;
RUN;


TITLE2 'Cluscar Dataset (Partial Dataset)';
PROC PRINT Data=hcluspref (Obs=20);
RUN;

/*Running K means*/
proc fastclus data=preferences_pca replace=full maxc=4 
		REPLACE=FULL distance out=k4preferences_pca maxiter=50 ;
var prin1 -- prin7 booking;
id RespID;
run;

/*Getting cluseters on the original dataset*/

proc sql;
	create table clusters_pca as 
	select a.*, b.cluster
	from preferences_binary a
	inner join k4preferences_pca b
	on a.respid = b.respid
	order by cluster;
run;


proc sql;
	select cluster, count(*), sum(case when age >= 60 then 1 else 0 end)/count(*)
	from clusters_pca
	group by cluster
	order by cluster;
run;




PROC CANDISC data=k4preferences_pca Anova OUT=can_pca;
class cluster;
var prin1 -- prin9 booking;
run;

TITLE3 'Plot of canonical variables identified by cluster';
PROC SGPLOT data=can_pca;
Scatter y=can2 x=can1 /group=cluster;
RUN;

/*Analysis of Clusters*/

proc means data=clusters_pca n mean median std nmiss;
	var &varclus_variables age income income_miss trips male nov -- ethnicity_notanswer ;
	by cluster;
run;

/*********RUN TILL HERE*******/


/***Variable Clustering***/
/*Determining number of clusters*/
ods graphics on;
proc cluster data=preferences_stand method=ward simple noeigen RMSSTD RSQ PSEUDO CCC print=15 outtree=tree;
var &varclus_variables booking;
run;
ods graphics off;

/*Dendrogram of cluster solution*/
TITLE2 'Dendrogram of the Cluster Solution';
PROC TREE Data=tree levels=0.035 OUT=hcluspref ;;
RUN;
/*4 Clusters*/


proc fastclus data=preferences_stand replace=full maxc=4 REPLACE=FULL distance out=preferences_varclus maxiter=50 ;
var &varclus_variables booking;
id RespID;
run;

/*Getting cluseters on the original dataset*/

proc sql;
	create table clusters_varclus as 
	select a.*, b.cluster
	from preferences_binary a
	inner join preferences_varclus b
	on a.respid = b.respid
	order by cluster;
run;

PROC CANDISC data=preferences_varclus Anova OUT=can_varclus;
class cluster;
var &varclus_variables booking;
run;

TITLE3 'Plot of canonical variables identified by cluster';
PROC SGPLOT data=can_varclus;
Scatter y=can2 x=can1 /group=cluster;
RUN;

/*Analysis of Clusters*/

proc means data=clusters_pca n mean median std nmiss;
	var &varclus_variables age income income_miss trips male nov -- ethnicity_notanswer ;
	by cluster;
run;



















/*Sex*/
proc freq data=clusters ;
	tables sex*cluster /norow nopercent chisq;
run;

/*refreshing*/
ods html close; /* close previous */
ods html;
proc univariate data=clusters noprint;
	histogram refreshing;
	inset n mean median std / position=ne;
	title 'Reference';
run;
proc univariate data=clusters noprint;
	histogram refreshing;
	inset n mean median std / position=ne;
	by cluster;
	title 'Cluster';
run;


