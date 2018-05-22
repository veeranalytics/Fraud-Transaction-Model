# Count Time
start_time <- Sys.time()

# Load Libraries

library(plyr)
library(dplyr)
library(caret)
library(randomForest)
library(ggplot2)
library(stringr)
library(rattle)
library(pROC)

# Load Data
fraud_df <- read.csv("https://raw.githubusercontent.com/veeranalytics/Fraudulent-Transactions-Model/master/fraud_full_sample.csv")

# Does step have any clear pattern in distribution across Fraud cases?
# Note that step indicates the hour within the month that this data was captured 
# so these plots should be considered time-series.

# This brings the full dataset down to 2.8M records. 
# That is a reduction of 56% in the dataset which should eliminate a lot of noise.
# Create sample dataset
not_fraud <- fraud_df %>%
  filter(isFraud == "No") %>%
  sample_n(8213)

is_fraud <- fraud_df %>%
  filter(isFraud == "Yes")

# Plot chart for step
ggplot(fraud_df, aes(x = step, col = isFraud, fill = isFraud)) + 
  geom_histogram(bins = 743) + 
  scale_fill_manual(values=c("#04B431", "#FA5858")) +
  ggtitle("Comparison of Fraud Vs Non-Fraud Trasactions") +
  xlab("Hours Elapsed Since Start of Month") +
  ylab("Frequency")

# Look at only Fraud cases.
# Plot chart for step
ggplot(is_fraud, aes(x = step)) + 
  geom_histogram(bins = 743) +
  ggtitle("Histogram of Fraudulent Transaction based on Timing") +
  xlab("Hours Elapsed Since Start of Month") +
  ylab("Frequency")

## Plots for Continous variables
#  
#df$category <- cut(df$a, breaks=c(-Inf, 0.5, 0.6, Inf), labels=c("low","middle","high"))
#res <- df %>% mutate(category=cut(a, breaks=c(-Inf, 0.5, 0.6, Inf), labels=c("low","middle","high"))split(das, cut(das$anim, 3)

# chart_df <- fraud_df                                                                                                         
# chart_df <- split(das, cut(das$anim, 3))


# Plot chart for Amount
ggplot(fraud_df, aes(x = amount, col = isFraud, fill = isFraud)) + 
  geom_histogram(bins = 743) +
  scale_fill_manual(values=c("#04B431", "#FA5858")) +
  ggtitle("Transaction by Amount") +
  xlab("Amount") +
  ylab("Frequency")

# Plot chart for oldbalanceOrg
ggplot(fraud_df, aes(x = oldbalanceOrg, col = isFraud, fill = isFraud)) + 
  geom_histogram(bins = 743) +
  scale_fill_manual(values=c("#04B431", "#FA5858")) +
  ggtitle("Transaction by Old Balance At Origin") +
  xlab("Balance") +
  ylab("Frequency") + 
  xlim(0, 50000) +
  ylim(0, 50)

# Plot chart for newbalanceOrig
ggplot(fraud_df, aes(x = newbalanceOrig, col = isFraud, fill = isFraud)) + 
  geom_histogram(bins = 743) +
  scale_fill_manual(values=c("#04B431", "#FA5858")) +
  ggtitle("Transaction by New Balance At Origin") +
  xlab("Balance") +
  ylab("Frequency") +
  xlim(0, 50000) +
  ylim(0, 5)

# Plot chart for oldbalanceDest
ggplot(fraud_df, aes(x = oldbalanceDest, col = isFraud, fill = isFraud)) + 
  geom_histogram(bins = 743) +
  scale_fill_manual(values=c("#04B431", "#FA5858")) +
  ggtitle("Transaction by Old Balance At Destination") +
  xlab("Balance") +
  ylab("Frequency") +
  xlim(0, 50000) +
  ylim(0, 5)

# Plot chart for newbalanceDest
ggplot(fraud_df, aes(x = newbalanceDest, col = isFraud, fill = isFraud)) + 
  geom_histogram(bins = 743) +
  scale_fill_manual(values=c("#04B431", "#FA5858")) +
  ggtitle("Transaction by New Balance At Destination") +
  xlab("Balance") +
  ylab("Frequency") +
  xlim(0, 50000) +
  ylim(0, 5)

# Stacked barplot with for transaction Type
type_df <- aggregate(step ~ type + isFraud, data = fraud_df, FUN = length)
type_df1 <- group_by(df, type) %>% 
        mutate(percent = step/sum(step))

ggplot(data=type_df1, aes(x=type, y=percent, fill = isFraud)) +
  geom_bar(stat="identity") +
  scale_fill_manual(values=c("#04B431", "#FA5858")) +
  geom_text(aes(label=paste0(round(percent*100,1),"%")), col = "white",
            position = position_stack(vjust = 0.5), size = 4) +
  theme_minimal() +
  ggtitle("Fraudulent Transactions by Transaction Type") +
  xlab("Transaction Type") +
  ylab("Percent")

# Pre-processing the full dataset for modelling
preproc_model <- preProcess(fraud_df[, -1], 
                            method = c("center", "scale", "nzv"))

fraud_preproc <- predict(preproc_model, newdata = fraud_df[, -1])

# Remove high correlation columns
model_df <- fraud_preproc %>%
  select(-newbalanceDest)

# Bind the results to the pre-processed data
model_df <- cbind(isFraud = fraud_df$isFraud, model_df)

# Split sample into train and test sets
in_train <- createDataPartition(y = model_df$isFraud, p = .75, 
                                list = FALSE) 
train <- model_df[in_train, ] 
test <- model_df[-in_train, ] 

# Create Model
# Set general parameters
# Create control used to fit all models
# We will use three iterations of 10-fold cross-validation 
# for every model so that we can compare apples-to-apples.
control <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        repeats = 3, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

# Random Forest model
grid <- expand.grid(.mtry = 5, .ntree = seq(25, 150, by = 25))

rf_model <- train(isFraud ~ ., 
                  data = train, 
                  method="rf", 
                  metric = "ROC", 
                  TuneGrid = grid, 
                  trControl=control)

# print(rf_model$finalModel)
# plot(rf_model$finalModel)

# Plot variable importance
varImpPlot(rf_model$finalModel)

# Predict on Training set
rf_train_pred <- predict(rf_model, train)
confusionMatrix(train$isFraud, rf_train_pred, positive = "Yes")

# Predict on Test set
rf_test_pred <- predict(rf_model, test)
confusionMatrix(test$isFraud, rf_test_pred, positive = "Yes")
