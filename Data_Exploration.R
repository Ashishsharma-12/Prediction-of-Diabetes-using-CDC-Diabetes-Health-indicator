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
  ggplot2,
  gridExtra
)

# Load Dataset from UCI
# CDC Diabetes Health Indicators 
# UCI Machine Learning Repository. 


data <-  import("./datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv") %>% as_tibble() 

df <- data

dim(df)
head(df)
str(df)

# unique values in each column
# checking the binary, ordinal and numeric features in the dataset

apply(df, 2, function(x) length(unique(x)))

# Proportion of target classes

round(prop.table(table(df$Diabetes_binary)) * 100, digits = 1)

# Check if any na values

apply(df, MARGIN = 2, FUN = function(x) sum(is.na(x)))
na.omit(df)

# Check if any duplicates

sum(duplicated(df))

# Using base R functions to find duplicate values

duplicates <- df[duplicated(df), ]

duplicates

# proptable for duplicates

round(prop.table(table(duplicates$Diabetes_binary)) * 100, digits = 1)

# remove duplicated rows

library(sqldf)

df <- sqldf('SELECT DISTINCT * FROM df')
dim(df)
round(prop.table(table(df$Diabetes_binary)) * 100, digits = 1)
glimpse(df)

target <- df['Diabetes_binary']
features_binary <- c('HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                   'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 
                   'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex')
features_ordinal <- c('GenHlth', 'Age', 'Education', 'Income')
features_numerical <- c('BMI', 'MentHlth', 'PhysHlth')

# ``````````````````````````````````````````````
# Visualization
# ``````````````````````````````````````````````

clr <- target %>% 
  rename(y = Diabetes_binary) %>%   
    mutate(
      y = ifelse(
        y == 0,
        "Healthy",
        "Diabetes"
      )
    )  %>%
  pull(y) %>% as_factor

target %>% 
  select(Diabetes_binary) %>% 
  ggplot( aes(x = Diabetes_binary, fill = clr)) +
  geom_bar(stat="count",alpha= .5, position="dodge") +
  labs(title = "Distribution of Target", x='Classes', y='Count') +
  theme_minimal()

# ~~~~~~~~~~~~~~~~Plotting all binary feature~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# crete emplty list to store plot
plots <- list()

# Loop through each binary feature and create subplots
for (i in 1:length(features_binary)) {
  
  feature_data <- df[, c(features_binary[i]), drop = FALSE]  # Ensure it's a dataframe
  
  p <- ggplot(feature_data, aes(x = .data[[features_binary[i]]], col= clr)) +
    geom_bar() +
    ggtitle(paste("Distribution of", features_binary[i]))
  
  plots[[i]] <- p
}

# Arrange plots into a grid layout
grid.arrange(grobs = plots, nrow = 4, ncol = 4)

# ~~~~~~~~~~~~~~~~Plotting all ordinal feature~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# crete emplty list to store plot
plots <- list()

# Loop through each ORDINAL feature and create subplots
for (i in 1:length(features_ordinal)) {
  
  feature_data <- df[, c(features_ordinal[i]), drop = FALSE]  # Ensure it's a dataframe
  
  p <- ggplot(feature_data, aes(x = .data[[features_ordinal[i]]], col = clr)) +
    geom_histogram() +
    ggtitle(paste("Distribution of", features_ordinal[i]))
  
  plots[[i]] <- p
}

# Arrange plots into a grid layout
grid.arrange(grobs = plots, nrow = 2, ncol = 2)

# ~~~~~~~~~~~~~~~~Plotting all numerical feature~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# crete emplty list to store plot
plots <- list()

# Loop through each ORDINAL feature and create subplots
j <- 4
for (i in 1:length(features_numerical)) {
  
  feature_data <- df[, c(features_numerical[i]), drop = FALSE]  # Ensure it's a dataframe
  
  p <- ggplot(feature_data, aes(x = .data[[features_numerical[i]]], col = clr)) +
    geom_histogram() +
    ggtitle(paste("Distribution of", features_numerical[i]))
  
  plots[[i]] <- p
  
  q <- ggplot(feature_data, aes(x = .data[[features_numerical[i]]])) +
    geom_boxplot() +
    ggtitle(paste("Distribution of", features_numerical[i]))
  
  plots[[j]] <- q
  j <- j + 1
  
}

# Arrange plots into a grid layout
grid.arrange(grobs = plots, nrow = 2, ncol = 3)

# ~~~~~~~~~~~~~~~~~~~~~~~~~ Outliers detection for numerical features

boxplot(df[features_numerical])

#``````````````````````````````````````````
# Outlier removal for BMI
#``````````````````````````````````````````

boxplot(df[features_numerical[1]])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~  Min-Max


mm <- apply(df[features_numerical[1]], MARGIN = 2, FUN = function(x) (x - min(x))/diff(range(x)))
boxplot(mm, main = "min_max")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~  Scale

sc <- scale(df[features_numerical[1]])
boxplot(sc, main = "Scale")

# ~~~~~~~~~~~~~~~~~~ Z-Score

z1 <- apply(df[features_numerical[1]], MARGIN = 2, FUN = function(x) (x - mean(x))/sd(x))
z2 <- apply(df[features_numerical[1]], MARGIN = 2, FUN = function(x) (x - mean(x))/(2*sd(x)))

boxplot (z1, main = "Z-score, 1 sd")
boxplot (z2, main = "Z-score, 2 sd")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~  Softmax
library(DMwR2)

sm <- apply(df[features_numerical[1]], MARGIN = 2, FUN = function(x) (SoftMax(x,lambda = 1, mean(x), sd(x))))
boxplot (sm, main = "Soft Max for BMI, Softmax")

df[features_numerical[1]] <- sm

#``````````````````````````````````````````
# Outlier removal for MentHlth
#``````````````````````````````````````````

boxplot(df[features_numerical[2]])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~  Softmax
library(DMwR2)

sm1 <- apply(df[features_numerical[2]], MARGIN = 2, FUN = function(x) (SoftMax(x,lambda = 1, mean(x), sd(x))))
boxplot (sm, main = "Soft Max for BMI MentHlth, Softmax, lambda = 1")

df[features_numerical[2]] <- sm1

#``````````````````````````````````````````
# Outlier removal for PhysHlth
# ``````````````````````````````````````````

boxplot(df[features_numerical[3]])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~  Softmax

sm <- apply(df[features_numerical[3]], MARGIN = 2, FUN = function(x) (SoftMax(x,lambda = 1, mean(x), sd(x))))
boxplot (sm, main = "Soft Max for BMI, Softmax")

df[features_numerical[3]] <- sm

rm(sm1,mm,sc,z1,z2)

sm <- apply(df[features_numerical], MARGIN = 2, FUN = function(x) (SoftMax(x,lambda = 1,mean(x), sd(x))))
boxplot (sm, main = "Soft Max for Numeric variable, Softmax lambda= 1")

df[features_numerical] <- sm

# ~~~~~~~~~~~~~~~~Plotting all numerical feature after outliers removal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# crete emplty list to store plot
plots <- list()

# Loop through each ORDINAL feature and create subplots
j <- 4
for (i in 1:length(features_numerical)) {
  
  feature_data <- df[, c(features_numerical[i]), drop = FALSE]  # Ensure it's a dataframe
  
  p <- ggplot(feature_data, aes(x = .data[[features_numerical[i]]], col = clr, alpha = 0.5)) +
    geom_histogram() +
    ggtitle(paste("Distribution of", features_numerical[i]))
  
  plots[[i]] <- p
  
  q <- ggplot(feature_data, aes(x = .data[[features_numerical[i]]])) +
    geom_boxplot() +
    ggtitle(paste("Distribution of", features_numerical[i]))
  
  plots[[j]] <- q
  j <- j + 1
  
}

# Arrange plots into a grid layout
grid.arrange(grobs = plots, nrow = 2, ncol = 3)

# ~~~~~~~~~~~~~~~~~Calculate the correlation matrix~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(corrplot)

# Plot the correlation matrix for ordinal variables

corr <- cor(df[features_ordinal])
corrplot(corr, method = "number", title="Correlation matrix")

# cor plot with y

corr <- cor(df[features_ordinal], df[1], method = c("pearson"))
corrplot(corr, method = "number", title="Correlation matrix with target")
# ------------------------------------------------------------------------------------
# Plot the correlation matrix for numeric variables

corr <- cor(df[features_numerical])
corrplot(corr, method = "number", title="Correlation matrix")

# cor plot with y

corr <- cor(df[features_numerical],df[1], method = c("pearson"))
corrplot(corr, method = "number", title="Correlation matrix with target")
# ------------------------------------------------------------------------------------
# Plot the correlation matrix for binary variables

corr <- cor(df[features_binary])
corrplot(corr, method = "pie", title="Correlation matrix", type = c("upper"))

# cor plot with y

corr <- cor(df[features_binary], df[1], method = c("pearson"))
corrplot(corr, method = "number", title="Correlation matrix with target")
# ------------------------------------------------------------------------------------

# Changing 0,1 into Labels, and Rename class variable as `y` and changing it into a factor variable

# df %<>%
#   rename(y = Diabetes_binary) %>%
#   mutate(
#     y = ifelse(
#       y == 0,
#       "Healthy",
#       "Diabetes"
#     )
#   )

df %<>%
  rename(y = Diabetes_binary) %>% 
  mutate(
    y = ifelse(
      y == 0,
      "Healthy",
      "Diabetes"
    )
  ) %>% 
 as_factor()


# Splitting data into Train test samples using train_test_split()
p_load(creditmodel)

?train_test_split

train_test = train_test_split(df,split_type = "Random", prop = 0.8,seed = 12, save_data = FALSE)

trn = train_test$train
tst= train_test$test

dim(trn)
dim(tst)

round(prop.table(table(trn$y)) * 100, digits = 1)
round(prop.table(table(tst$y)) * 100, digits = 1)

df  %>% saveRDS("./datasets/dataset.RDS")
trn %>% saveRDS("./datasets/Train_data.RDS")
tst %>% saveRDS("./datasets/Test_data.RDS")



rm(list = ls())

dev.off()












