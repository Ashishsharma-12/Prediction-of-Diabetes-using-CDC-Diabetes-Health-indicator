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

df  <- readRDS("datasets/dataset.RDS")
trn <- readRDS("datasets/Train_data.RDS")
tst <- readRDS("datasets/Test_data.RDS")

# trn_y <- trn %>% pull(y) %>% as_factor()
tst_y <- tst %>% pull(y) %>% as_factor()

# trn <- trn %>% select(-y)
# 
tst <- tst %>% select(-y)


# ````````````````````````
# Modelling
# ````````````````````````

library(class)
# ?ceiling
k = ceiling(sqrt(55246))

if(k %% 2==0){
  k = k+1
} 

# ?knn
dim(trn)
length(trn_y)

preds <- knn(train = trn, test = tst, cl = trn_y, k=k)

preds

length(preds)

length(tst_y)

library(gmodels)
# look at help for gmodels and CrossTable

# Create the cross tabulation of predicted vs. actual
CrossTable(x = tst_y, y = preds, prop.chisq=FALSE)

# Confusion Matrix

cm <- confusionMatrix(preds, tst_y )

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# RESULTS on test data ##################################################

# kNN Model
# Accuracy: 0.7336  
# Reference: Healthy: 4562     1452
# Reference: Diabetes: 2227     5570
# Sensitivity = 0.6720   
# Specificity = 0.7932   


# Ensembled model: COMPUTE KNN MODEL ON TRAINING DATA USING REPEATED CV #######################

# Define parameters for kNN
statctrl <- trainControl(
  method  = "repeatedcv",  # Repeated cross-validation
  number  = 5,              # Number of folds
  repeats = 3               # Number of sets of folds
)  

# Set up parameters to try while training (3-19)
k <- seq(37, 61, by = 2)

# Apply model to training data

runtime <- system.time({
  fit <- train(
    y ~ ., 
    data = trn,             
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
