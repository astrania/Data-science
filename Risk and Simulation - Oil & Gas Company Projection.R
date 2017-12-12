library(graphics)
library(haven)
library(ks)
library(scales)
library(MASS)
library(readxl)
library(triangle)
library(ggplot2)
library(ggplot2)

######## assignment #############
# Calculate the 90% VaR of the NPV distribution for Oil and its 90% Bootstrapped Confidence Interval.
#  Calculate the 90% Conditional VaR for Oil (or expected shortfall) and it’s 90% Bootstrapped Confidence Interval.
### Assumptions###########
# The following assumptions should be made in this RFP.
#  The drilling cost should be a fixed cost.(this means we shouldn't be scaling with units)
#  When calculating drilling cost, use geometric returns from 1991-2006 and project to 2018 (12 years).
#  Be sure to use exp(R) in your drilling cost projections (since these are geometric rates).
#  Be sure to identify if you are using a normality assumption or if you are using the kernel density estimator when projecting drilling cost.

setwd("/Desktop/MSA DATA/risk and simulation")
price <- read_excel('Analysis_Data.xlsx', skip = 2)  # read in prices 
cost <- read_excel('Analysis_Data.xlsx', skip = 2, sheet = 2)[2:47,c(1,2,5)] #read in cost
price= as.data.frame(price)
cost = as.data.frame(cost[31:46,]) #select geometric returns from 1991 -2006
cost= as.numeric(cost[,3])
cost = exp(cost)

# Process##
# we are using the simple equation (profit= Price* units - cost) however we are trading profit for net present value(npv) 
# Npv is a fancy way of saying "if we invest in the project for x years what can we expect our profit to be
# so now we have ( npv=  Price* units - cost)  we need to create distributions for all the inputs and produce the distribution NPV as a an output
# once we have npv we can look at the distribution and pull the Value at risk (VAR) and the espected shortfall (ES)
#step 1(#units)   create distribution of units produced, we did this by using a triangle distribution. rtriangle() is the function
#step 2 (#price)  build price distribution, we first want to check for normality of historical values
# step 3 ( cost) build cost distribution ( choose between sim a norm or kernal - do a qqplot to decide)
# step 4 plug all your distributions in to get npv= units*price - cost
# step 5 monte carlo simulation to do using a bootstrap to simulate all possibilities of a Var 
#units##
# build unit per year triangle distribution, Values given, min= 10000 max=17000 mean=15500

simulation.size=10000 #set simulation size to 10,000 (seems like a solid number)
units = rtriangle(simulation.size, a=10000, b=17000, c=15500) #triangle distribution function a=min b=max c=mode
hist(units)
# check to see what price projection errors look normal


#price##
qqnorm(price[,4]) 
qqline(price[,4], col = 2)  #build qq plot
#according to this price projections are normal, this means we can simmulated off of these
#build price distribution based on projections
price.projection = rnorm(n=10000,mean = (price[,4]), sd=  sd(price[,4])) #using inbeded functions to pull the mean and stdev from the price projections to build distribution 
hist(price.projection)
##cost##
#lets check cost distribution to see if it's normal

# resign column for simplicity 

# Because of the normality of the qq plot we should use normal dist over kernal 
qqnorm(cost) 
qqline(cost, col = 2) # looks nice and normal
c.mean = mean(cost) #pull mean
c.sd = sd(cost) # pull stdev
#build normal cost distribution going out 12 years 
cost.matrix=matrix((rnorm(simulation.size, mean = c.mean, sd = c.sd)),n=12*simulation.size,nrow=12) #simulate cost values matrix, 12 rows of 1000 possible values 
cost.prediction=2238600*apply(cost.matrix,2,prod)  # 2238600 is the cost for oil in 2006 . (excel sheet values in thousands)

hist(cost.prediction)
quantile(cost.prediction,.5)


##calculate Net present value###
####idrees##
Units <- matrix(rep(0, 23*simulation.size), nrow = 23)
Price <- matrix(rep(0, 23*simulation.size), nrow = 23)
for (i in 1:23){
  # cost.prediction[i,]=2238600*apply(cost.matrix,2,prod)
  Units[i,] <- rtriangle(simulation.size, a=10000, b=17000, c=15500)
  Price[i,] <- rtriangle(simulation.size, a=as.numeric(priceproj[i+1,3]), b=as.numeric(priceproj[i+1,2]), c=as.numeric(priceproj[i+1,4]))
}
# qplot(cost.prediction[1,], main= "Cost for drilling oil well in 2018", ylab= 'Count',xlab= "Cost($)", fill=I("blue"),
#       col=I("red"),
#       alpha=I(.2)) # builds matrix 1 + random )
# median(net.value.present)
rev1= Price*Units
total.rev= apply(rev1,2,sum)
npv= total.rev- cost.prediction
hist(npv)
hist(npv)

net.value.norm <- Price*Units - cost.prediction
net.value.present <- apply(net.value.norm,2,sum)

net.value.annual <- apply(net.value.norm,1,mean)

hist(net.value.norm)
####
# revenue1= units*price.projection
# r.mean= revenue1
# r.sd = sd(revenue1)
# revenue.matrix=matrix((rnorm(simulation.size, mean = r.mean, sd = r.sd)),n=22*simulation.size,nrow=22)
# 
# apply(revenue.matrix)
# 
# npv= revenue.matrix - cost.prediction
# 
# npv = ((units*price.projection)- cost.prediction)/1000  # units * Price - fixed cost ( Units are in 1000 so it's easier to read)
# hist(npv)
# median(npv) # very low, very negitive 
# step 5 Monte Carlo Simulation #
n.simulations <- 100000
VaR.percentile = 0.10 # because we are looking at a 90% var we want to look at thebottom %.10  
VaR <- quantile(npv, VaR.percentile, na.rm=TRUE)  #find value at risk looking at the quantile 
VaR.label <- dollar(VaR) # change units to dollars
VaR
#graph it
hist(npv, main='Net Present Value for 2018', xlab='Value Change', col="lightblue")
breaks = c(-20, -10, 0, 10, 20)



abline(v = VaR, col="red", lwd=2) # set var line
mtext(paste("Value at Risk",VaR.label), col="red") # color 
ES <- mean(npv[npv < VaR], na.rm=TRUE) #looking at the mean inside of the VAR
dollar(ES) # units in $1000
#assume boot strap for calculation of VAR
n.bootstraps = 1000 # arbitrary number selection 
sample.size = 1000 # arbitrary number selection 

VaR.boot <- rep(0,n.bootstraps) #init
ES.boot <- rep(0,n.bootstraps)
for(i in 1:n.bootstraps){
  bootstrap.sample <- sample(npv, size=sample.size, replace = T) #bootstrap is sample with replacing, taking 10,000 bos and sample 1000 at a var of 10% 
  VaR.boot[i] <- quantile(bootstrap.sample, VaR.percentile, na.rm=TRUE)  # calulate all siulations var and save
  ES.boot[i] <- mean(bootstrap.sample[bootstrap.sample < VaR.boot[i]], na.rm=TRUE) #calculate es which is the mean of everything less than the VAR line
}
VaR.boot.U <- quantile(VaR.boot, 0.95, na.rm=TRUE) #creating your confidence interval upper
VaR.boot.L <- quantile(VaR.boot, 0.5, na.rm=TRUE) # creating confidence interval lower
dollar(VaR.boot.L) #convert to dollar format  and give lower bound
dollar(VaR) #convert to dollar and give var
dollar(VaR.boot.U) #convert to dollar and give upper 


hist(npv/1000000,  main='Net Present Value for 2018', xlab='Value Change', col="lightblue")
# 
breaks = c(-20, -10, 0, 10, 20)
abline(v = VaR/1000000, col="red", lwd=3) # red line on Value ate risk 
mtext(paste("Value at Risk","17.3M", sep=" = "), at = VaR/1000000, col="red")

abline(v = VaR.boot.L/1000000, col="blue", lwd=2, lty="dashed")  #blue dashed line on lower conf. limit
abline(v = VaR.boot.U/1000000, col="blue", lwd=2, lty="dashed") # bue dashed line on upper 
cv=npv
es.boot.U <- quantile(ES.boot, 0.95, na.rm=TRUE) #creating your confidence interval upper for es
es.boot.L <- quantile(ES.boot, 0.05, na.rm=TRUE) # creating confidence interval lower es
dollar(es.boot.L) #convert to dollar format  and give lower bound
dollar(es) #convert to dollar and give var
dollar(VaR.boot.U) #convert to dollar and give upper 


ESG = npv[npv < VaR]
hist(ESG/1000000, breaks=50, main='Expected Shortfall', xlab='Value Change', col="lightblue")
breaks = c(-20, -10, 0, 10, 20)
abline(v = ES/1000000, col="red", lwd=3) # red line on Value ate risk 
mtext(paste("Expected ShortFall",'$11.1m', sep=" = "), at = VaR/1200000, col="red")

abline(v = es.boot.L/1000000, col="blue", lwd=2, lty="dashed")  #blue dashed line on lower conf. limit

abline(v = es.boot.U/1000000, col="blue", lwd=2, lty="dashed") # bue dashed line on upper 

write.csv(cv,"pv.csv")
VaR.boot.L