protected void updateParameters(long userID, long itemID, float rating, double currentLearningRate) {
    int userIndex = userIndex(userID);
    int itemIndex = itemIndex(itemID);

    double[] userVector = userVectors[userIndex];
    double[] itemVector = itemVectors[itemIndex];
    double prediction = predictRating(userIndex, itemIndex);
    double err = rating - prediction;

    // adjust user bias
    userVector[USER_BIAS_INDEX] +=
        biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * userVector[USER_BIAS_INDEX]);

    // adjust item bias
    itemVector[ITEM_BIAS_INDEX] +=
        biasLearningRate * currentLearningRate * (err - biasReg * preventOverfitting * itemVector[ITEM_BIAS_INDEX]);

    // adjust features
    for (int feature = FEATURE_OFFSET; feature < numFeatures; feature++) {
      double userFeature = userVector[feature];
      double itemFeature = itemVector[feature];

      double deltaUserFeature = err * itemFeature - preventOverfitting * userFeature;
      userVector[feature] += currentLearningRate * deltaUserFeature;

      double deltaItemFeature = err * userFeature - preventOverfitting * itemFeature;
      itemVector[feature] += currentLearningRate * deltaItemFeature;
    }
  }
