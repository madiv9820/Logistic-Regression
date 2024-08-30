import gradient_descent
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()

X = dataset['data']
Y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y)

algo = gradient_descent.LogisticRegression()
algo.fit(x_train, y_train)

print('Score: ', algo.score(x_train, y_train))