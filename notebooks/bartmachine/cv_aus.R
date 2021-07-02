# Title     : TODO
# Objective : TODO
# Created by: mamu867
# Created on: 4/29/21
#dev.off(dev.list()["RStudioGD"])

library(caret)

df <- read.csv(
    '/Users/mamu867/PNNL_Mac/PNNL_Code_Base/UQ_AL/data/aus.csv')
CT_RT <- df$CT_RT
df <- within(df, rm(CT_RT))
set.seed(readline("What is the value of seed?")) 
test_inds = createDataPartition(y = 1:length(CT_RT), p = 0.2, list = F) 

df_test = df[test_inds, ] 
CT_RT_test = CT_RT[test_inds] 
df_train = df[-test_inds, ]
CT_RT_train = CT_RT[-test_inds]

##build BART regression model
options(java.parameters="-Xmx5000m")
library(bartMachine)
bart_machine <- bartMachine(df_train, CT_RT_train, num_trees=1000, seed=42)

plot_y_vs_yhat(bart_machine, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine, prediction_intervals = TRUE)
plot_y_vs_yhat(bart_machine, Xtest=df_test, ytest=CT_RT_test, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine, Xtest=df_test, ytest=CT_RT_test, prediction_intervals = TRUE)

rsq <- function(x, y) summary(lm(y~x))$r.squared
y_pred <- predict(bart_machine, df_test)
rsq(CT_RT_test, y_pred)

cor.test(CT_RT_test, y_pred, method=c("pearson"))

