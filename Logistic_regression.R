setwd(dirname(file.choose()))
getwd()

pacman::p_load(  # Use p_load function from pacman
  caret,         # Train/test functions
  e1071,         # Machine learning functions
  magrittr,      # Pipes
  pacman,        # Load/unload packages
  rattle,        # Pretty plot for decision trees
  rio,           # Import/export data
  tidyverse,      # So many reasons
  cluster,
  car,
  ggplot2,
  GGally,
  dplyr,
  MASS,
  pROC
)

set.seed(1)

df  <- import("datasets/dataset.RDS")
trn <- import("datasets/Train_data.RDS")
tst <- import("datasets/Test_data.RDS")

trn %<>%
  mutate(
    y = ifelse(
      y == "Healthy",
      0,
      1
    )
  )


tst %<>%
  mutate(
    y = ifelse(
      y == "Healthy",
      0,
      1
    )
  )

model <- glm(y~.,family = binomial, data = trn)

summary(model) #57453

?stepAIC

model_2 <- stepAIC(model)

summary(model_2) #57450

summary(model_2$fitted.values)

hist(model_2$fitted.values,main = " Histogram ",xlab = "Probability of 'pos' diabetes", col = 'light green')

preds <- ifelse(model_2$fitted.values >0.5,1,0)
pred <-  ifelse(model$fitted.values >0.5,1,0)

# Evaluation

model$aic
model_2$aic

# confusion table

mytable_1 <- table(trn$y,preds)
rownames(mytable_1) <- c("Act. neg","Act. pos")
colnames(mytable_1) <- c("Pred. neg","Pred. pos")
mytable

accuracy_1 <- sum(diag(mytable_1))/sum(mytable_1)
accuracy_1

mytable_2 <- table(trn$y,pred)
rownames(mytable_2) <- c("Act. neg","Act. pos")
colnames(mytable_2) <- c("Pred. neg","Pred. pos")
mytable

accuracy_2 <- sum(diag(mytable_2))/sum(mytable_2)
accuracy_2

# ROC Curve

library(ROCR)

predictions <- predict(model, type = "response")
predictions
# Create a prediction object for ROCR
prediction_objects <- prediction(predictions, trn$y)
prediction_objects
# Create an ROC curve object
roc_object <- performance(prediction_objects, measure = "tpr", x.measure = "fpr")

# Plot the ROC curve
plot(roc_object, main = "ROC Curve", col = "blue", lwd = 2)

# Add labels and a legend to the plot
legend("bottomright", legend = 
         paste("AUC =", round(performance(prediction_objects, measure = "auc")
                              @y.values[[1]], 2)), col = "blue", lwd = 2)

auc(y~model_2$fitted.values, data = trn)
# --------------------------------------------------------------------------------
# accuracy on tst data

predict_y <- predict(model_2,tst, type = "response")
predict_y
predict_y <- ifelse(predict_y >0.5,1,0)
predict_y

tst_table <- table(tst$y, predict_y)
accuracy <- sum(diag(tst_table))/sum(tst_table)
accuracy


ROCPred <- prediction(predict_y, tst$y)
ROCPer <- performance(ROCPred, measure = "tpr", x.measure = "fpr")

auc <- performance(ROCPred, measure = "auc")
auc <- auc@y.values[[1]]
auc

# Plotting curve
plot(ROCPer)
plot(ROCPer, colorize = TRUE,
     print.cutoffs.at = seq(0.1, by = 0.1),
     main = "ROC CURVE")
abline(a = 0, b = 1)

auc <- round(auc, 4)
legend(.6, .4, auc, title = "AUC", cex = 1)
