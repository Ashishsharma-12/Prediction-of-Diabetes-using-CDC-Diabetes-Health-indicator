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
  tidymodels,
  car,
  ggplot2,
  GGally,
  dplyr,
  MASS,
  pROC
)

set.seed(1)

tidymodels_prefer()

df  <- import("datasets/dataset.RDS")
trn <- import("datasets/Train_data.RDS")
tst <- import("datasets/Test_data.RDS")

trn_y <- trn %>% pull(y) %>% as_factor()
tst_y <- tst %>% pull(y) %>% as_factor()

trn <- trn[2:22]
tst <- tst[2:22]

# -------------------------------------------
# Modelling with glm engine
# -------------------------------------------

logreg_cls_spec <- logistic_reg() %>% 
  set_engine("glm")

logreg_cls_spec

logreg_cls_fit <- logreg_cls_spec %>% fit(trn_y ~ ., data = trn)

logreg_cls_fit

# ---------------------- accruracy  on train -----------------------------------------

preds <- predict(logreg_cls_fit, trn)
# Get the confusion matrix
cm <- confusionMatrix(data = preds$.pred_class, reference = trn_y, positive= '1')

# Print the confusion matrix
cm %>% print()

# Accuracy on train data: 74.35

# ---------------------- accruracy  on test -----------------------------------------

preds <- predict(logreg_cls_fit, tst)
# preds
# Get the confusion matrix
cm <- preds$.pred_class %>%
  confusionMatrix(reference = tst_y,positive= '1')

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy on test data: 73.56
# Sensitivity : 0.7632         
# Specificity : 0.7072 

#     0    1
# 0 4801 1663
# 1 1988 5359

# ---------------------------------------------------------------------------------


# -------------------------------------------
# Modelling with ensemble techniques
# -------------------------------------------

statctrl <- trainControl(
      method  = "repeatedcv",  # Repeated cross-validation
      number  = 5,              # Number of folds
      repeats = 3               # Number of sets of folds
    )

logreg_ens_cls_fit <- train(y ~ ., data = trn, method = "glm", trControl = statctrl)
logreg_ens_cls_fit
# ---------------------- accruracy  on train -----------------------------------------

preds <- predict(logreg_ens_cls_fit, trn)
preds
preds_class <- ifelse(preds >= 0.5, 1, 0)
preds_class <- factor(preds_class, levels = levels(trn_y))

# Get the confusion matrix
cm <- confusionMatrix(data = preds_class, reference = trn_y)

# Print the confusion matrix
cm %>% print()

# Accuracy on train data: 74.22

# ---------------------- accruracy  on test -----------------------------------------

preds <- predict(logreg_ens_cls_fit, tst)
preds
preds_class <- ifelse(preds >= 0.5, 1, 0)
preds_class <- factor(preds_class, levels = levels(trn_y))
# preds
# Get the confusion matrix
cm <- preds_class %>%
  confusionMatrix(reference = tst_y, positive='1')

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy on test data: 73.59
# Sensitivity : 0.7710          
# Specificity : 0.6997 

#     0    1
# 0 4750 1608
# 1 2039 5414

# ---------------------------------------------------------------------------------