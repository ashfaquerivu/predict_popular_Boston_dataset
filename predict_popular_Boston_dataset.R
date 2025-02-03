install.packages(c("caret", "randomForest", "ggplot2"))
library(caret)
library(randomForest)
library(ggplot2)

# Load dataset
set.seed(123)
data <- MASS::Boston  # Load the Boston dataset from MASS package

# Check for missing values
sum(is.na(data))

# Split into training and testing
trainIndex <- createDataPartition(data$medv, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Feature Scaling and Normalization
preProc <- preProcess(trainData, method = c("center", "scale"))
trainData <- predict(preProc, trainData)
testData <- predict(preProc, testData)

# Train Random Forest model
set.seed(123)
rf_model <- train(medv ~ ., data = trainData, method = "rf",
                  trControl = trainControl(method = "cv", number = 10),
                  tuneLength = 5)

# Print best tuning parameters
print(rf_model$bestTune)

# Make predictions
testPred <- predict(rf_model, testData)

# Evaluate Model Performance
rmse <- sqrt(mean((testPred - testData$medv)^2))
rsq <- cor(testPred, testData$medv)^2

cat("Root Mean Squared Error:", rmse, "\n")
cat("R-squared:", rsq, "\n")

# Feature Importance
importance <- varImp(rf_model, scale = FALSE)
print(importance)

# Plot Feature Importance
plot(importance)