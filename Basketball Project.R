set.seed(7)

#importing libraries

if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, caret, glmnetUtils, corrplot, pROC)

#importing the dataset

basket <- read.csv('student_2614867data.csv')

#printing the structure of the dataset

str(basket)

#removing irrelevant columns

basket$X <- NULL
basket$Name <- NULL
basket$Yrs <- NULL

basket$Year_drafted <- as.factor(basket$Year_drafted)
basket$Target <- as.factor(basket$Target)

#splitting the dataset into a training and test set

trainIndex <- createDataPartition(basket$Target, p = 0.7, 
                                  list = FALSE)
train.set <- basket[ trainIndex,]
test.set  <- basket[-trainIndex,]

#scaling & centering the data

scalingData <- preProcess(train.set[,-21],
                          method = c("center", "scale"))
train.set[,-21] <- predict(scalingData,train.set[,-21])
test.set[,-21] <- predict(scalingData,test.set[,-21])

#printing the dimensions of the training & test set

dim(train.set)
dim(test.set)

#plotting the distribution of the target

ggplot(data=train.set, aes(x=Target, col=Target))+ 
  geom_bar(stat="count", fill='white')+
  labs(title='Target Distribution - NBA Dataset', 
       x='0 if career<=5 yrs, 1 otherwise')

#printing a proportion table

table(train.set$Target)

#drawing a correlation plot

corrplot(cor(train.set[,- c(1,21)]))

#formatting the train set for glmnet

x <- model.matrix(Target~.-1, train.set)
y <- train.set$Target
x.test <- model.matrix(Target ~.-1, test.set)

#fitting a lasso model

cv.lasso <- cv.glmnet(x, y, alpha=1, type.measure='class',
                      family='binomial')

#printing outputs

cv.lasso
coef(cv.lasso)

#computing predictions - lasso models

pred.lasso.min <- predict(cv.lasso, x.test, s=cv.lasso$lambda.min, 
                         type='class')
pred.lasso.1se <- predict(cv.lasso, x.test, s=cv.lasso$lambda.1se, 
                      type='class')

#computing metrics - lasso models

#Lasso Model with lambda.min
confusionMatrix(as.factor(pred.lasso.min), test.set$Target, positive='1')
#Lasso Model with lambda.1se
confusionMatrix(as.factor(pred.lasso.1se), test.set$Target, positive='1')

#drawing ROC curve - lasso models

plot.roc(test.set$Target, as.numeric(pred.lasso.min), 
         main='Lasso w/ lambda.min')

plot.roc(test.set$Target, as.numeric(pred.lasso.1se), 
         main='Lasso w/ lambda.1se')

#fitting a ridge model

cv.ridge <- cv.glmnet(x, y, alpha=0, type.measure='class', 
                      family='binomial')
#printing outputs

cv.ridge
coef(cv.ridge)

#computing predictions - ridge model

pred.ridge.min <- predict(cv.ridge, x.test, s=cv.ridge$lambda.min, 
                          type='class')
pred.ridge.1se <- predict(cv.ridge, x.test, s=cv.ridge$lambda.1se, 
                          type='class')

#computing metrics - ridge models

#Ridge model w/ lambda.min
confusionMatrix(as.factor(pred.ridge.min), test.set$Target, positive='1')
#Ridge model w/lambda.1se
confusionMatrix(as.factor(pred.ridge.1se), test.set$Target, positive='1')

#drawing ROC curve - Ridge models

plot.roc(test.set$Target, as.numeric(pred.ridge.min), 
         main='Ridge w/ lambda.min')

plot.roc(test.set$Target, as.numeric(pred.ridge.1se), 
         main='Ridge w/ lambda.1se')

#finding alpha for the elastic net model

cv.net <- cva.glmnet(x, y, type.measure='class', 
                     family='binomial', alpha=seq(0,1,0.1))

#plotting best alpha

plot(cv.net)

cv.net <- cv.glmnet(x, y, type.measure='class', 
                    family='binomial', alpha=0.1)

#computing predictions - elastic net

pred.net.min<- predict(cv.net, x.test, s=cv.net$lambda.min, 
                          type='class')
pred.net.1se <- predict(cv.net, x.test, s=cv.net$lambda.1se, 
                          type='class')

#computing metrics - elastic net

#Elastic Net w/ lambda.min
confusionMatrix(as.factor(pred.net.min), test.set$Target, positive='1')
#Elastic Net w/ lambda.1se
confusionMatrix(as.factor(pred.net.1se), test.set$Target, positive='1')

#drawing ROC curve - elastic nets

plot.roc(test.set$Target, as.numeric(pred.net.min), 
         main='Elastic Net w/ lambda.min')

plot.roc(test.set$Target, as.numeric(pred.net.1se), 
         main='Elastic Net w/ lambda.1se')
