#Load libraries
library(broom)
library(caret)
library(doParallel)
library(ggplot2)
library(tidyverse)
library(lubridate)
library(rpart.plot)
library(lime)

#Set up parallel processing
cl <- makeCluster(detectCores()-4)
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

#PCA
prin_comp <- prcomp(train.m[,-53], scale. = T)
biplot(prin_comp, scale = 0)

library(ggbiplot)
g <- ggbiplot(prin_comp, obs.scale = 1, var.scale = 1, 
              ellipse = TRUE, 
              circle = TRUE)

saveRDS(g,"biplot.rds")

std_dev <- prin_comp$sdev
pr_var <- std_dev^2
pr_var[1:10]
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:20]
prop_varex <- as.data.frame(prop_varex)

scree <- ggplot(data = prop_varex, aes(x=1:52, y=cumsum(prop_varex))) +
  geom_point() + geom_line() + 
  xlab("Principal Component") +
  ylab("Cumulative Proportion of Variance Explained")

saveRDS(scree,"scree.rds")

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
saveRDS(train.rf, "rfmodel.rds")

#Test LIME to interpret the RF model
lime.data <- test.m[sample(nrow(test.m), 3), ]
rf.lime <- lime(train.m, train.rf, bin_continuous = TRUE, n_bins = 5, n_permutations = 1000)
rf.lime.explain <- explain(lime.data, rf.lime, n_labels=3,n_features=5)
rf.lime.plot <- plot_features(rf.lime.explain, ncol = 3)
saveRDS(rf.lime.plot, "rflimeplot.rds")

#Gradient boosted trees
date()
train.gbm <- train(classe ~ ., data=train.m, method="gbm", metric=metric, trControl=control, tuneLength=5, verbose = FALSE)
print(train.gbm)
plot(train.gbm)
date()
saveRDS(train.gbm, "gbtmodel.rds")

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

#Gaussian Process model - trying this model out of curiousity for a future project
gp.data <- train.m[sample(nrow(train.m), 2000), ]

date()
train.gpr <- train(classe ~ ., data=train.m, method="gaussprRadial", metric=metric, trControl=control, tuneLength=5, verbose = FALSE)
print(train.gpr)
plot(train.gpr)
date()
saveRDS(train.gpr, "gprmodel.rds")

#MLP - adding a simple neural network with LIME for model interpretability to test both libraries
train.m$classe <- as.factor(train.m$classe)

mlp_grid = expand.grid(layer1 = c(50),
                       layer2 = c(50),
                       layer3 = c(50))

date()
train.mlp <- train(classe ~ ., data=train.m, method="mlpML", 
                   preProc =  c('center', 'scale', 'knnImpute', 'pca'),
                   metric=metric, trControl=control,
                   tuneGrid = mlp_grid,
                   verbose = FALSE)
print(train.mlp)
plot(train.mlp)
date()
saveRDS(train.mlp, "mlpmodel.rds")

#Test LIME to interpret the MLP model
mlp.lime <- lime(train.m, train.mlp, bin_continuous = TRUE, n_bins = 5, n_permutations = 1000)
mlp.lime.explain <- explain(lime.data, mlp.lime, n_labels=3,n_features=5)
mlp.lime.plot <- plot_features(mlp.lime.explain, ncol = 3)
saveRDS(mlp.lime.plot, "mlplimeplot.rds")

#Compare models within test CV
summary(results)

caret.perf.plot <- function(caret.models){ #Takes a list of caret models and compares their metrics in a plot
  #Add code to check if all list objects are caret models
  results <- resamples(test) 
  scales <- list(x=list(relation="free"), y=list(relation="free"))
  return(bwplot(results, scales=scales))
}

model.comparison <- list(CART=train.rpart, RF=train.rf, GBM=train.gbm, SM=train.mn, SVM=train.svm)
caret.perf.plot(model.comparison)

#Most important variables in the RF
varImp(train.rf)

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

#Test RF and GBM on held-out test set
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
