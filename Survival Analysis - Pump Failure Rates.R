# Survial analysis homework #3
#  Data was collected from 770 pump stations. There are five potential failure conditions which
#  take the following values in the variable reason:
#  (0) No failure
#  (1) Flood – overflow or accumulation of water that submerges the pump station
#  (2) Motor – mechanical failure
#  (3) Surge – onshore gush of water typically associated with levee or structural damage
#  (4) Jammed – accumulation of trash or landslide materials

# There are eight factors that may influence the survivability of the pump stations.Not all
# pumps  have  each  characteristic, but  some  are  available  for  maintenance  or  upgrade  and
# denoted as such:
# Backup pump (upgrade available) – a redundant system used to protect the station
# from flooding when the main pump is not operating
# Bridge crane (upgrade available) – allows vertical access to equipment and protecting materials
# Servo (upgrade available) – servomechanism used to provide control of a desired operation through the Supervisory Control and Data Acquisition (SCADA) systems
# Trash-rack cleaner (upgrade available) – protects hydraulic structures against the inlet
# of debris, vegetation, or trash
# Elevation (maintenance available) – elevation of the pump station; may be altered by
# one foot through maintenance
# Slope – ravine slope surrounding the pump station
# Age – difference between the pump’s installation and the current date
# H1–H48  –  pumping  status  reported  by  pump  stations  during  a  48-hour  emergency
# period (accuracy of pump status not guaranteed to be error free)

#Provide  a  follow-up  to  your  last  report  and  a  set  of  recommendations  summarizing  the
# findings from your analysis.  This new report should include the following information:
# In  this  assignment,  you  will  model motor and surge failures  together  and  treat  all
# other failure reasons as censored.
# 1) Create both an AFT model and a Cox regression model with the following variables:
# backup, bridgecrane, servo, trashrack, elevation, slope, age
# Which of these models do you prefer?
# 2) Provide the coefficient estimates and standard errors from your Cox regression model
# and interpret some of them (at least one categorical and one continuous). Is there any
# evidence that any of these effects might not be constant over time?
# 3) The Army Corps of Engineers believes that motor failure is more likely if the motor
# has been running for 12 consecutive hours prior to failure.  Add this to your model
# using the H1--H48variables and describe its effect on failure.  What is your conclusion?

##############################
# Survival homework-3
library(survival)
library(haven)
library(flexsurv)
library(survminer)
library(ranger)
library(ggplot2)
library(tidyverse)
library(ggfortify)

# load data
data_dir <- "Fall 3\\Survival Analysis\\"
input_file <- "katrina.sas7bdat"
katrina <- read_sas(paste(data_dir, input_file, sep = ""))

# focus only on motor and surge(reason == c(2, 3)), others are treated as censored
# fit AFT models using weibull distribution, flexsurvreg option
# display coefficient estimates
# visualize result
fit.aft <- flexsurvreg(Surv(hour, reason == c(2,3)) ~ backup + bridgecrane + servo + 
                   trashrack + elevation + slope + age, data = katrina, dist = "weibull")
summary(fit.aft)

surv <- survreg(formula = Surv(hour, reason == c(2,3)) ~ backup + bridgecrane + servo + 
                  trashrack + elevation + slope + age, data = katrina, dist = "weibull")
summary(surv)

plot(fit.aft, type = "survival",ci = TRUE, conf.int = FALSE, 
     xlab = "Hour", ylab = "Survival Rate",ylim = c(0.8,1), main = "AFT Model")
plot(fit.aft, type = "cumhaz",ci = TRUE, conf.int = FALSE, 
     xlab = "Hour", ylab = "Hazard Rate", main = "Cumulative Hazard Model")

# fit cox regression model using coxph()
# display coefficient estimates
# coefficient estimate interpretation: see cox regression slide 17
# visualize result
# concordance: 0.671
fit.cox <- coxph(Surv(time = hour, event = reason %in% c(2,3))  ~ backup + bridgecrane + servo + 
                   trashrack + elevation + slope +
                   age, data = katrina)
summary(fit.cox)
autoplot(survfit(fit.cox), main = "Cox Regression Proportional Hazard Model", xlab = "Hour", ylab = "Survival rate",
         ylim = c(0.8,1), xlim = c(0,48))
survConcordance(Surv(time = hour, event = reason %in% c(2,3)) ~ predict(fit.cox), data = katrina)

## conclusion: pick cox model due to better fit to data
##             but need to further verify if cox's model assumptions are satified

# checking Cox's assumption: linearity, using martingale residual
# martingale residuals vs. time
m.res <- residuals(fit.cox, type = "martingale")
plot(katrina$hour, m.res, pch = 19, cex = 0.5, xlab = "Hour", 
     ylab = "Martingale Residuals", main = "Martingale Residuals vs. Time")
cens <- which(katrina$reason != c(2,3))
uncens <- which(katrina$reason == c(2,3))
points(katrina$hour[cens], m.res[cens], pch = 19, cex = 0.5, col = "blue")
points(katrina$hour[uncens], m.res[uncens], pch = 19, cex = 0.5, col = "red")
legend("bottomleft", inset = 0.05, c("Censored", "Failed"), col = c("blue", "red"),
       pch = 19)
abline(h = 0)

# check linearity, the only continuous predictor is "age"
### conclusion: linearity assumption satified(lowess line is pretty linear)
plot(katrina$age, m.res, pch = 19, cex = 0.5, xlab = "Age",
     ylab = "Martingale Residuals", main = "Testing Continous Variable Age for Linearity")
lines(lowess(katrina$age, m.res), col = "red")

# checking Cox's assumption: PH
# testing correlation of residuals with time
## conclusion: PH assumption failed to be satified(ideally the line should be straight)
##             in particular, "age" might not be constant over time(time-variant coefficient)
cox.zph(fit.cox)
plot(cox.zph(fit.cox), main = "")

# build a new Cox model
# that satisfy "motor has been running for 12 consecutive hours prior to failure"
# which is h1==1, up to h12==1, using variables h1-h48
## conclusion; surprisingly, survival rate is higher for motor that have been running 12hrs prior failure
fit.cox.motor_running <- coxph(Surv(time = hour, event = reason %in% c(2,3)) ~ backup + bridgecrane + servo + 
                                 trashrack + elevation + slope +
                                 age + (h1 == 1)  + (h2 == 1) + (h3 == 1) + (h4 == 1) + (h5 == 1) + (h6 == 1) + (h7 == 1) + (h8 == 1) + (h9 == 1) + (h10 == 1)
                               + (h11 == 1) + (h12 == 1) + h13 + h14 + h15 + h16 + h17 + h18 + h19 + h20
                               + h21 + h22 + h23 + h24 + h25 + h26 + h27 + h28 + h29 + h30
                               + h31 + h32 + h33 + h34 + h35 + h36 + h37 + h38 + h39 + h40
                               + h41 + h42 + h43 + h44 + h45 + h46 + h47 + h48, data = katrina)
autoplot(survfit(fit.cox.motor_running), main = "Cox Proportional Hazard Model for Motors Running 12hrs Prior", xlab = "Hour", ylab = "Survival rate",
         ylim = c(0.8,1), xlim= c(0,48))