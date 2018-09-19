# we need dplyr and e1071 for SVM/RF
library(dplyr)
library(e1071)
library(MASS)
# envoke RF library
library(randomForest)
# Regression/classification tree library
library(rpart)
# plotting trees
library(rpart.plot)
# enable caret for CV
library(caret)

# set a random seed for reproducbility (not during CV though...)
set.seed(321)

### change the following line to point to your CSV file:
trees=read.csv("E:\\ENVS 316\\assignment 5\\covtype_sm_sample.csv",row.names=1)
# correct factor variables:
trees$Type=as.factor(trees$Type)
trees$Area1[trees$Area1==1]=1 #as.factor(trees$Area1)
trees$Area2[trees$Area2==1]=2 #as.factor(trees$Area1)
trees$Area3[trees$Area3==1]=3 #as.factor(trees$Area1)
trees$Area4[trees$Area4==1]=4 #as.factor(trees$Area1)
trees$Area=as.factor(trees$Area1+trees$Area2+trees$Area3+trees$Area4)
trees[,11:14]<-NULL
# work with a random sample subset of the data to speed things up:
trees.sub <- trees[sample(1:nrow(trees), 1000, replace=FALSE),]

# Problem here: because we are doing subsetting, we end up with empty classes.
# Remove rare groups (n<10) from data:
trees.sub.rm <- trees.sub %>% group_by(Type) %>% filter(n() >= 10)
# expunge empty levels from factor Type
trees.sub.rm$Type=factor(trees.sub.rm$Type)
View(trees.sub.rm)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Students need to start editing the script here ####
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  fit RF to Type on all predictor vars:
# ntree = number of decision trees to grown, mtry = number of subset variables per split
fit <- randomForest(Type ~ .,data = trees.sub.rm, ntree = 5000,mtry = 2)
# make predictions of Group and test accuracy (*not* cross-validated)
pred=predict(fit,newdata=trees.sub.rm,type="class")
confusionMatrix(trees.sub.rm$Type,pred)$overall
#out of bag error
fit

#using train function from caret
train_control <- trainControl(method = "repeatedcv",number=10,repeats=2) 
tune_grid = expand.grid(mtry=c(3,4,5))
cv_fit <- train(Type ~.,data = trees.sub.rm,method="rf",trControl=train_control,tuneGrid=tune_grid,ntree=5000)
cv_fit
plot(varImp(cv_fit))

#SVM
C=10^(-1:5)
G=2^seq(-15,3,2)
ncv<-10 # number of cross-validation folds

# tune SVM using train() under caret
tune_grid <- expand.grid(C = C,sigma=G)
train_control <- trainControl(method = "cv",number=ncv)   

system.time(svm_fit<- train(Type~.,data=trees.sub.rm,
                            method = "svmRadial",   # Choose kernel?
                            tuneGrid = tune_grid,
                            #tuneLength = ?,  # number of combinations of tuning parameter(s)
                            trControl=train_control)) #,epsilon = epsilon
svm_fit
plot(svm_fit)

#try other gamma value
G=2^seq(3,9,2)?seq
tune_grid <- expand.grid(C = C,sigma=G)
train_control <- trainControl(method = "cv",number=ncv)   

system.time(svm_fit<- train(Type~.,data=trees.sub.rm,
                            method = "svmRadial",   # Choose kernel?
                            tuneGrid = tune_grid,
                            #tuneLength = ?,  # number of combinations of tuning parameter(s)
                            trControl=train_control)) #,epsilon = epsilon
svm_fit
plot(svm_fit)
?randomForest

