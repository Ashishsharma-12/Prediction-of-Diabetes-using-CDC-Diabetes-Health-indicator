setwd(dirname(file.choose()))
getwd()

pacman::p_load(  # Use p_load function from pacman
  caret,         # Train/test functions
  e1071,         # Machine learning functions
  magrittr,      # Pipes
  pacman,        # Load/unload packages
  tidyverse,      # So many reasons
  cluster,
  randomForest,
  ranger,
  tidymodels
)

set.seed(1)

tidymodels_prefer()

df  <- readRDS("datasets/dataset.RDS")
trn <- readRDS("datasets/Train_data.RDS")
tst <- readRDS("datasets/Test_data.RDS")

trn_y <- trn %>% pull(y) %>% as_factor()
tst_y <- tst %>% pull(y) %>% as_factor()

trn <- trn %>% select(-y)

tst <- tst %>% select(-y)
# -------------------------------------------
# Modelling
# -------------------------------------------
?rand_forest

rf_cls_spec <-rand_forest(trees = 500) %>% # This model can be used for classification or regression, so set mode
  set_mode("classification") %>% 
  set_engine("randomForest")

rf_cls_spec

rf_cls_fit <- rf_cls_spec %>% fit(trn_y ~ ., data = trn)

rf_cls_fit

# ---------------------- accruracy  on train -----------------------------------------

preds <- predict(rf_cls_fit, trn)
# Get the confusion matrix
cm <- confusionMatrix(data= preds$.pred_class, reference = trn_y)

# Print the confusion matrix
cm %>% print()

# Accuracy on train data: 94.84

# ---------------------- accruracy  on test -----------------------------------------

preds <- predict(rf_cls_fit, tst)
# preds
# Get the confusion matrix
cm <- preds$.pred_class %>%
  confusionMatrix(reference = tst_y, positive = '1')

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy on test data: 74.04
# Sensitivity : 0.7483        
# Specificity : 0.6964

#     0    1
# 0 4728 1524
# 1 2061 5498

# ----------------------- Tunning model -----------------------

?tuneRF

mtry <- tuneRF(trn,trn_y, ntreeTry=500, stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m) # 2

#  model
rf_tuned_cls_spec <-rand_forest(trees = 1000, mtry = 2) %>% 
  set_mode("classification") %>% 
  set_engine("randomForest")

rf_tuned_cls_spec

rf_tuned_cls_spec_fit <- rf_tuned_cls_spec %>% fit(trn_y ~ ., data = trn)

rf_tuned_cls_spec_fit

# ---------------------- accruracy  on train -----------------------------------------

preds <- predict(rf_tuned_cls_spec_fit, trn)
# preds
# Get the confusion matrix
cm <- confusionMatrix(data = preds$.pred_class, reference = trn_y)

# Print the confusion matrix
cm %>% print()

# Accuracy on train data: 80.17

# ---------------------- accruracy  on test -----------------------------------------

preds <- predict(rf_tuned_cls_spec_fit, tst)
# preds
# Get the confusion matrix
cm <- preds$.pred_class %>%
  confusionMatrix(reference = tst_y, positive = '1')

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy on test data: 74.19
# Sensitivity : 0.7830          
# Specificity : 0.6960  

#     0    1
# 0 4725 1524
# 1 2064 5498

# ----------------------------------------------
?train

statctrl <- trainControl(method="repeatedcv", search="grid")
tunegrid <- expand.grid(.mtry=c(1:3))

runtime <- system.time({
  fit <- train(
    trn_y ~ ., 
    data = trn,             
    method = "rf",  
    metrics = "Accuracy",
    trControl = statctrl,   
    tuneGrid = tunegrid,
    ntrees = 500,
  )
}) #this took ____________ min

print(runtime)
?randomForest
# ---------------------- accruracy  on train -----------------------------------------

preds <- predict(fit, trn)
preds
# Get the confusion matrix
cm <- confusionMatrix(data = preds$.pred_class, reference = trn_y)

# Print the confusion matrix
cm %>% print()

# Accuracy on train data: 79.81

# ---------------------- accruracy  on test -----------------------------------------

preds <- predict(fit, tst)
# preds
# Get the confusion matrix
cm <- preds$.pred_class %>%
  confusionMatrix(reference = tst_y, positive = '1')

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy on test data: 73.97
# Sensitivity : 0.6958          
# Specificity : 0.7821 

#     0    1
# 0 4710 1524
# 1 2079 5498










