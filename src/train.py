from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """Train and return a Logistic Regression classifier"""
    model= LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model