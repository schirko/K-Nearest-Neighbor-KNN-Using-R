# Load required libraries
library(caret) # needed for dummyVar function
library(class) # needed for knn function
library(gmodels) # needed for CrossTable function

rm(list=ls()) #remove base objects to clear workspace

# 1.1  Data exploration and preprosessing
heart_data <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data") #import heart data
str(heart_data)

# Need to give the columns names
colnames(heart_data) <- c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","num")

# Check for and remove NA values
is.na(heart_data)
df <- data.frame(heart_data)
which(is.na(df))
sum(is.na(df))
colSums(is.na(df))
df_nonas <- na.omit(df)

# Convert all nums <= 1 as 1 (with heart disease)
df_nonas$num <- with(df_nonas, ifelse(num >=1, 1, 0)) 
df_nonas$num <- as.factor(df_nonas$num) 

# Normalize data
min_max_norm <- function(x) {(x-min(x)) / (max(x) - min(x))}
df_norm <- as.data.frame(lapply(df_nonas[,c(1,4,5,8,10)], min_max_norm))
cat_vars <- as.data.frame(lapply(df_nonas[,c(3,7,11:13)],as.factor))

# 1.2 Create training set w/ 70:30, predict w/ 1-N, give error rate
dummy <- dummyVars(~.,data=cat_vars,fullRank = TRUE) #convert factors to dummy variables
df_dummy <- as.data.frame(predict(dummy,newdata=cat_vars)) 
df_all <- cbind(df_nonas$sex,df_nonas$fbs,df_norm,df_dummy,df_nonas$num) 
set.seed(789)  #set seed for random generator

# Predict w/ 1-N, give error rate
idx <- sample(2, nrow(df_all), replace=TRUE, prob=c(0.7, 0.3)) #create 2 samples with 70:30
df_train <- df_all[idx==1, ] #sample 1 for the train set
df_test <- df_all[idx==2,]   # sample 2 for the test set
df_train_label <- df_all[idx==1,ncol(df_train)] #labels for train set - ncol is the last column will be used in knn()
df_test_label <- df_all[idx==2,ncol(df_test)] # labels for test set - will be used to compare with the predicted values from the model  
knn_pred.1 <- knn(train=df_train,test=df_test,cl=df_train_label,k=17) # train with 1 neighbor

# Evaluate the model performance
CrossTable(x=knn_pred.1, y=df_test_label, prop.chisq=FALSE)


