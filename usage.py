from classifier import SimpleNetwork

clf = SimpleNetwork()

X_train = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
]

y_train = [
    1,
    0,
    0,
    1,
]

clf.fit(X_train, y_train)

print(clf.predict([[1, 1, 0]]))
