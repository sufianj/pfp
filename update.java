protected void update(Preference preference, double mu) {
    int userIndex = userIndex(preference.getUserID());
    int itemIndex = itemIndex(preference.getItemID());

    double[] userVector = userVectors[userIndex];
    double[] itemVector = itemVectors[itemIndex];

    double prediction = dot(userVector, itemVector);
    double err = preference.getValue() - prediction;

    // adjust features
    for (int k = FEATURE_OFFSET; k < rank; k++) {
      double userFeature = userVector[k];
      double itemFeature = itemVector[k];

      userVector[k] += mu * (err * itemFeature - lambda * userFeature);
      itemVector[k] += mu * (err * userFeature - lambda * itemFeature);
    }

    // adjust user and item bias
    userVector[USER_BIAS_INDEX] += biasMuRatio * mu * (err - biasLambdaRatio * lambda * userVector[USER_BIAS_INDEX]);
    itemVector[ITEM_BIAS_INDEX] += biasMuRatio * mu * (err - biasLambdaRatio * lambda * itemVector[ITEM_BIAS_INDEX]);
    //System.out.printf("for (u=%d, i=%d) userbias = %f, itembias = %f \n",userIndex, itemIndex, userVector[USER_BIAS_INDEX],itemVector[ITEM_BIAS_INDEX] );
  }
