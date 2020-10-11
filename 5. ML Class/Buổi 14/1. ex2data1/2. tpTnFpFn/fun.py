def tpTnFpFn(y_test, y_pred, positiveLabel):
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    for i in range(y_test.shape[0]):
        if y_test[i] == positiveLabel and y_pred[i] == positiveLabel:
            truePositive = truePositive + 1

        elif y_test[i] != positiveLabel and y_pred[i] != positiveLabel:
            trueNegative = trueNegative + 1

        elif y_test[i] != positiveLabel and y_pred[i] == positiveLabel:
            falsePositive = falsePositive + 1

        elif y_test[i] == positiveLabel and y_pred[i] != positiveLabel:
            falseNegative = falseNegative + 1
    return truePositive, trueNegative, falsePositive, falseNegative