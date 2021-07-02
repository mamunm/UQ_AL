# Title     : TODO
# Objective : TODO
# Created by: mamu867
# Created on: 4/29/21

library(caret)

df <- read.csv(
    '/Users/mamu867/PNNL_Mac/PNNL_Code_Base/UQ_AL/data/chrome.csv')
CT_RT <- df$CT_RT
df <- within(df, rm(CT_RT))
set.seed(42) 
test_inds = createDataPartition(y = 1:length(CT_RT), p = 0.2, list = F) 

df_test = df[test_inds, ] 
CT_RT_test = CT_RT[test_inds] 
df_train = df[-test_inds, ]
CT_RT_train = CT_RT[-test_inds]


library(ggplot2)
ggplot2::qplot(log(CT_RT), 
      geom="histogram",
      #binwidth=0.1,
      main="Histogram of RT",
      xlab="CT_RT",
      fill=I("green"),
      col=I("black"))

library(corrplot)
correlations <- cor(df)
corrplot::corrplot(correlations, method="circle", order="hclust")

##build BART regression model
options(java.parameters="-Xmx5000m")
library(bartMachine)
bart_machine = bartMachine(df_train, CT_RT_train)
summary(bart_machine)

rmse_by_num_trees(bart_machine, 
                  tree_list=c(seq(5, 50, by=5)),
                  num_replicates=5)

bart_machine <- bartMachine(df_train, CT_RT_train, num_trees=40, seed=42)
plot_convergence_diagnostics(bart_machine)

check_bart_error_assumptions(bart_machine)

var_selection_by_permute(bart_machine, num_reps_for_avg=20)

#k_fold_cv(df, CT_RT, k_folds = 10)

plot_y_vs_yhat(bart_machine, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine, prediction_intervals = TRUE)
plot_y_vs_yhat(bart_machine, X_test=df_test, y_test=CT_RT_test, credible_intervals = TRUE)
plot_y_vs_yhat(bart_machine, X_test=df_test, y_test=CT_RT_test, prediction_intervals = TRUE)

investigate_var_importance(bart_machine, num_replicates_for_avg = 20)


interaction_investigator(bart_machine, num_replicates_for_avg = 25,
                         num_var_plot = 10, bottom_margin = 5)






