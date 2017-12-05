#Load libraries
library(broom)
library(caret)
library(doParallel)
library(ggplot2)
library(tidyverse)
library(lubridate)
library(rpart.plot)

#Set up parallel processing
cl <- makeCluster(8)
registerDoParallel(cl)
#stopCluster(cl)

#Read and explore data
setwd("~/GitHub/coursera-practical-machine-learning")
pml.train <- read_csv("pml-training.csv", trim_ws = TRUE)[,-1]
str(pml.train)
summary(pml.train)
sum(complete.cases(pml.train)) #Only 324 cases have complete data
colSums(is.na(pml.train)) #Data is almost entirely missing from most columns, making them pretty much useless
pml.train <- pml.train[,colMeans(is.na(pml.train)) < .9] #Remove aforementioned columns
pml.train <- pml.train[complete.cases(pml.train),] #One case is incomplete and dropped
pml.train <- select(pml.train, -c(1:6)) #First few columns look like metadata - nothing to do with the movement
nearZeroVar(pml.train ,saveMetrics=TRUE) #Check for remaining near zero variance columns - none left, so all fields are retained

#Create data sets and training settings
set.seed(555)
metric <- "Accuracy" #Metric of measurement for the assignment
#control <- trainControl(method="repeatedcv", number=10, repeats=3, summaryFunction = multiClassSummary, classProbs = TRUE) #lots of options, but only added classProbs and multiClass Summary
control <- trainControl(method="cv", number=3, summaryFunction = multiClassSummary, classProbs = TRUE) #faster cv for interim models
split.m <- createDataPartition(pml.train$classe, p = 0.7, list=FALSE)
train.m <- pml.train[split.m,]
test.m <- pml.train[-split.m,]

#Run models
#Decision tree
date()
train.rpart <- train(classe ~ ., data=train.m, method="rpart", metric=metric, trControl=control, tuneLength=5)
print(train.rpart)
plot(train.rpart)
date()

#Random forest
date() #Timestamp to view how long the model took to run
train.rf <- train(classe ~ ., data=train.m, method="rf", metric=metric, trControl=control, tuneLength=5)
print(train.rf) #Print model results
plot(train.rf) #Plot accuracy under different tuning parameters
date()

#Gradient boosted trees
date()
train.gbm <- train(classe ~ ., data=train.m, method="gbm", metric=metric, trControl=control, tuneLength=5, verbose = FALSE)
print(train.gbm)
plot(train.gbm)
date()

#Softmax
date()
train.mn <- train(classe ~ ., data=train.m, method="multinom", metric=metric, trControl=control, tuneLength=5, maxit = 500, verbose = FALSE)
print(train.mn)
plot(train.mn)
date()

#SVM
date()
train.svm <- train(classe ~ ., data=train.m, method="svmLinear", metric=metric, trControl=control, tuneLength=5, verbose = FALSE)
print(train.svm)
plot(train.svm)
date()

#Compare models within test CV - make a function of this
results <- resamples(list(CART=train.rpart, RF=train.rf, GBM=train.gbm, SM=train.mn, SVM=train.svm)) 
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

#Most important variables in the RF - make a function of this
MIV.l <- varImp(train.rf)
MIV.l$Variables <- row.names(MIV.l)
MIV.l <- MIV.l[order(-MIV.l$Overall),]
print(head(MIV.l))
MIV.l <- varImp(train.rf)
MIV.l$Variables <- row.names(MIV.l)
MIV.l <- MIV.l[order(-MIV.l$Overall),]
print(head(MIV.l))

#Confusion matrix for initial test data
train.rpart.predict <- predict(train.rpart, newdata = test.m)
train.rf.predict <- predict(train.rf, newdata = test.m)
train.gbm.predict <- predict(train.gbm, newdata = test.m)
train.mn.predict <- predict(train.mn, newdata = test.m)
confusionMatrix(train.rpart.predict, test.m$classe)
confusionMatrix(train.rf.predict, test.m$classe)
confusionMatrix(train.gbm.predict, test.m$classe)
confusionMatrix(train.nm.predict, test.m$classe)

#Predict classe for these for assignment
train.rpart.predict <- predict(train.rpart, newdata = pml.test)
train.rf.predict <- predict(train.rf, newdata = pml.test)
train.gbm.predict <- predict(train.gbm, newdata = pml.test)
train.mn.predict <- predict(train.mn, newdata = pml.test)
confusionMatrix(train.rpart.predict, pml.test$classe)
confusionMatrix(train.rf.predict, pml.test$classe)
confusionMatrix(train.gbm.predict, pml.test$classe)
confusionMatrix(train.nm.predict, pml.test$classe)

#Test RF only on held-out test set - this can go later
pml.test <- read_csv("pml-testing.csv", trim_ws = TRUE)[,-1]
predict(train.rf, pml.test)
predict(train.gbm, pml.test)

#Save objects for assignment write-up
saveRDS(bwplot(results, scales=scales), "model.comparison.plot.rds")

saveRDS(confusionMatrix(train.rf.predict, test.m$classe), "RFAcc.rds")
saveRDS(kable(confusionMatrix(train.rf.predict, test.m$classe)$table), "RFConf.rds")

saveRDS(confusionMatrix(train.gbm.predict, test.m$classe), "GBMAcc.rds")
saveRDS(kable(confusionMatrix(train.gbm.predict, test.m$classe)$table), "GBMConf.rds")

saveRDS(predict(train.rf, pml.test), "quiz.rds")
