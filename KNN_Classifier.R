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
  kknn,
  fpc
)

set.seed(1)

df  <- readRDS("datasets/dataset.RDS")
trn <- readRDS("datasets/Train_data.RDS")
tst <- readRDS("datasets/Test_data.RDS")

# Changing 0,1 into Labels, and Rename class variable as `y` and changing it into a factor variable

trn %<>%
  mutate(
    y = ifelse(
      y == 0,
      "Healthy",
      "Diabetes"
    )
  )

# Changing 0,1 into Labels, and Rename class variable as `y` and changing it into a factor variable

tst %<>%
  mutate(
    y = ifelse(
      y == 0,
      "Healthy",
      "Diabetes"
    )
  )

trn_y <- trn %>% pull(y) %>% as_factor()
tst_y <- tst %>% pull(y) %>% as_factor()

# trn <- trn %>% select(-y)
tst <- tst %>% select(-y)


# ````````````````````````
# Modelling
# ````````````````````````
?nearest_neighbor

knn_cls_spec <- nearest_neighbor(neighbors = 11, weight_func = "gaussian") %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_cls_spec


knn_cls_fit <- knn_cls_spec %>% fit(trn_y ~ ., data = trn)

knn_cls_fit

# ---------------------- accruracy  on train -----------------------------------------

preds <- predict(knn_cls_fit, trn)
preds
# Get the confusion matrix
cm <- confusionMatrix(data = preds$.pred_class, reference = trn_y)

# Print the confusion matrix
cm %>% print()

# Accuracy on train data: 82.49

# ---------------------- accruracy  on test -----------------------------------------

preds <- predict(knn_cls_fit, tst)
# preds
# Get the confusion matrix
cm <- preds$.pred_class %>%
  confusionMatrix(reference = tst_y)

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy on test data: 71.8
# Sensitivity : 0.6723          
# Specificity : 0.7622

#     0    1
# 0 4564  1670
# 1 2225  5352

# ---------------------------------------------------------------------------------
# optimal value of K using silhoutte coefficient
# ---------------------------------------------------------------------------------

best_k <- pamk(trn)
best_k
plot(pam(trn, best_k$nc))

# ---------------------------------------------------------------------------------
# Ensembled model: COMPUTE KNN MODEL ON TRAINING DATA USING REPEATED CV #######################
# ---------------------------------------------------------------------------------

# Define parameters for kNN
statctrl <- trainControl(
  method  = "repeatedcv",  # Repeated cross-validation
  # number  = 3,              # Number of folds
  # repeats = 3               # Number of sets of folds
)  

# Set up parameters to try while training (3-19)
k <- seq(3,50, by = 2)

# Apply model to training data

runtime <- system.time({
  fit <- train(
    tst_y ~ ., 
    data = tst,             
    method = "knn",         
    trControl = statctrl,   
    tuneGrid = data.frame(k)    
  )
}) #this took 21.219333 min

print(runtime)

# k = 58  accuracy= 0.7367291 kappa= 0.4724538 training accuracy

# Plot accuracy against various k values
fit %>% plot()                # Automatic range on Y axis
fit %>% plot(ylim = c(0, 1))  # Plot with 0-100% range

# Print the final model
fit %>% print() 

# APPLY MODEL TO TEST DATA #################################

# Predict test set
pred_knn_ensembled <- predict(    # Create new variable 
  fit,              # Apply saved model
  newdata = tst     # Use test data
)

# Get the confusion matrix
cm <- pred_knn_ensembled %>%
  confusionMatrix(reference = tst_y)

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# RESULTS on test data ##################################################

# kNN ensembled Model

# Accuracy: 0.7339  
# Reference: Healthy: 4574     1460
# Reference: Diabetes: 2215     5562
# Sensitivity = 0.6737   
# Specificity = 0.7921   
