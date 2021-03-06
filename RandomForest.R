library(rpart)
library(rpart.plot)
library(caTools)
library(DMwR)
rdata <- read.csv("/Users/deepak/Documents/fervez/creditcard.csv")
set.seed(1234)
rdata$Class <- factor(rdata$Class, levels=c(0,1))
split <- sample.split(rdata$Class, SplitRatio=0.7)
traindata <- subset(rdata, split=TRUE)
testdata <- subset(rdata, split=FALSE)
traindata$Class <- as.factor(traindata$Class)
traindata<- SMOTE(Class~., traindata, perc.over=200, perc.under=100)
str(data)
dt_rpart<-rpart(formula=Class~., data=traindata)
rpart.plot(dt_rpart)
summary(dt_rpart)
predict_target<- predict(dt_rpart, testdata, type="class")
xtab<-table(testdata$Class, predict_target, dnn=c("actual","predict"))
library(caret) 
confusionMatrix(xtab)
