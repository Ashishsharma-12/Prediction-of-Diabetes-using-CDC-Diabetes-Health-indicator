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
  tidymodels
)

set.seed(1)

df  <- readRDS("datasets/dataset.RDS")
trn <- readRDS("datasets/Train_data.RDS")
tst <- readRDS("datasets/Test_data.RDS")

glimpse(trn)

trn_y <- trn %>% pull(y) %>% as_factor()

tst_y <- tst %>% pull(y) %>% as_factor()

trn <- trn %>% select(-y)

tst <- tst %>% select(-y)
# -------------------------------------------
# Modelling
# -------------------------------------------

# Create a decision tree model specification
# ?decision_tree

show_engines("decision_tree")

r_tree <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification")


# Fit the model to the training data
fit <- r_tree %>%
  fit(trn_y ~ ., data = trn)

fit

# Load the library
library(rpart.plot)

# Plot the decision tree
rpart.plot(fit$fit, type = 4, extra = 101, under = TRUE, cex = 0.8, box.palette = "auto")

pred <- predict(fit, tst)
pred
pred <- factor(pred, levels = levels(tst_y))
length(tst_y)

# Get the confusion matrix
cm <- pred$.pred_class %>%
  confusionMatrix(reference = tst_y)

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy: 71.52

# -----------------------------------------------------

c_tree <- decision_tree() %>%
  set_engine("C5.0") %>%
  set_mode("classification")

# Fit the model to the training data
fit <- c_tree %>%
  fit(trn_y ~ ., data = trn)

fit

# Load the library
library(rpart.plot)

# Plot the decision tree
cpart.plot(fit$fit, type = 4, extra = 101, under = TRUE, cex = 0.8, box.palette = "auto")

pred <- predict(fit, tst)
pred
# pred <- factor(pred, levels = levels(tst_y))
# length(tst_y)

# Get the confusion matrix
cm <- pred$.pred_class %>%
  confusionMatrix(reference = tst_y)

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy: 73.28

# ------------------------------------------------
?boost_tree

boosted_tree <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = 10,
  trees = 1500,
  min_n = 2,
  tree_depth = 21,
  learn_rate = 0.01,
  loss_reduction = 1.5,
  sample_size = 0.75
)

boosted_tree

# Fit the model to the training data
fit <- boosted_tree %>%
  fit(trn_y ~ ., data = trn)

fit

pred <- predict(fit, tst)
pred
# pred <- factor(pred, levels = levels(tst_y))
# length(tst_y)

# Get the confusion matrix
cm <- pred$.pred_class %>%
  confusionMatrix(reference = tst_y, positive = '1')

# Plot the confusion matrix
cm$table %>% 
  fourfoldplot(color = c("red", "lightblue"))

# Print the confusion matrix
cm %>% print()

# Accuracy: 74.25
# Sensitivity : 0.7915          
# Specificity : 0.6917 



