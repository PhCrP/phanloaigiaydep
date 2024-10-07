from sklearn.metrics import hinge_loss, log_loss

def svm_loss_function(X_test, y_test, model):
    y_pred_decision = model.decision_function(X_test)
    loss = hinge_loss(y_test, y_pred_decision)
    print("SVM Loss:", loss)
    return loss

def mlp_loss_function(X_test, y_test, model):
    y_pred_prob = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred_prob)
    print("MLP Loss:", loss)
    return loss

def bagging_loss_function(X_test, y_test, models):
    
    svm_model = models['svm']
    cart_model = models['cart']

    y_pred_decision_svm = svm_model.decision_function(X_test)
    svm_loss = hinge_loss(y_test, y_pred_decision_svm)

    y_pred_prob_cart = cart_model.predict_proba(X_test)
    cart_loss = log_loss(y_test, y_pred_prob_cart)

    combined_loss = (svm_loss + cart_loss) / 2

    print("Bagging Loss:", combined_loss)

    return combined_loss
