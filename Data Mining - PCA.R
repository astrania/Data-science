# Principal Component Analysis on Leuk Error
# Dear Ms. Peceay,
# We are happy you reached out. Upon reviewing the data you provided to us, 
# we have some good news for you. After running principal components analysis, 
# our team identified some samples that look unusual. As you begin your 
# investigation of the possible mislabeled samples, we recommend that you 
# start with testing samples 19 and 2, respectively.
# Sample 19 is labeled as AML, but according to our analysis, it is most similar to the ALL-T samples. 
# Sample 2, labeled as AML, is similar to the ALL – B samples. 
# The mislabeled sample is likely to be one of those samples.
# We are confident that in following our recommendation, you will not only 
# save money and time, but will be able to rest assured that our procedures 
# should get you on the right track to meet your project delivery date on time.
# Please let us know if there is anything else we can assist you with.

library(tm)
load('/Mystery_Machine/Linear Algebra/HW1/LeukError.RData')
cats = leuk[,5001]
leuk = leuk[,1:5000]

pca = prcomp(leuk)
plot(pca$sdev^2)
plot(pca$x[,1:2], col=c("red","blue","green")[cats], pch=10)
legend(x='bottomright', c('ALL - B','ALL - T','AML'), pch='o', col=c('red','blue','green'), pt.cex=1.5)
labl = 1:38
labl[labl != 19 & labl != 2] = NA
text(pca$x[,1],pca$x[,2]+500,labl)

plot(pca$x[,c(1,3)], col=c("red","blue","green")[cats], pch=10)
legend(x='bottomright', c('ALL - B','ALL - T','AML'), pch='o', col=c('red','blue','green'), pt.cex=1.5)
labl = 1:38
labl[labl != 19 & labl != 2] = NA
text(pca$x[,1],pca$x[,3]+500,labl)

library(rgl)
library(car)
scatter3d(pca$x[,1],pca$x[,2],pca$x[,3],group = cats, pch = 10, surface = F)
