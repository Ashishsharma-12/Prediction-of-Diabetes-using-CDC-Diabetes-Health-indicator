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
  GGally
)

set.seed(1)

df  <- import("datasets/data.rds")
trn <- import("datasets/data_trn.rds")
tst <- import("datasets/data_tst.rds")
trn_y <- import("datasets/data_trn_y.rds")
tst_y <- import("datasets/data_tst_y.rds")

# ````````````````````````
# Modelling
# ````````````````````````

library(class)
?ceiling
k = ceiling(sqrt(920))

if(k %% 2==0){
  k = k+1
} 

?knn

pred_knn <- knn(train = trn, test = tst, cl = trn_y, k=k)

pred_knn

length(pred_knn)
length(tst_y)

library(gmodels)
# look at help for gmodels and CrossTable

# Create the cross tabulation of predicted vs. actual
CrossTable(x = tst_y, y = pred_knn, prop.chisq=FALSE)

# Confusion Matrix

cm <- confusionMatrix(pred_knn, tst_y )

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# RESULTS ##################################################

# kNN Model
# Accuracy: .
# Reference: Not Spam: 
# Reference: Spam: 
# Sensitivity = 
# Specificity = 


# COMPUTE KNN MODEL ON TRAINING DATA #######################

# Define parameters for kNN
statctrl <- trainControl(
  method  = "repeatedcv",  # Repeated cross-validation
  number  = 5,             # Number of folds
  repeats = 3              # Number of sets of folds
)  

# Set up parameters to try while training (3-19)
k = rep(seq(3, 20, by = 2), 2)

# Apply model to training data (takes a moment)
fit <- train(
  trn_y ~ ., 
  data = trn,                         # Use training data
  method = "knn",                     # kNN training method
  trControl = statctrl,               # Control parameters
  tuneGrid = data.frame(k),           # Search grid param.
  preProcess = c("center", "scale"),  # Preprocess
  na.action = "na.omit"
)

# Plot accuracy against various k values
fit %>% plot()                # Automatic range on Y axis
fit %>% plot(ylim = c(0, 1))  # Plot with 0-100% range

# Print the final model
fit %>% print()

# APPLY MODEL TO TEST DATA #################################

# Predict test set
pred_knn_ensembled <- predict(    # Create new variable ("predicted")
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

# RESULTS ##################################################

# kNN ensembled Model
# Accuracy: 
# Reference: Not Spam: 
# Reference: Spam: 
# Sensitivity = 
# Specificity = 
