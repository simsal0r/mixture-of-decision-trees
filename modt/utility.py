from sklearn import tree

def tree_accuracy(X, y, depth):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X, y)
    return(clf.score(X,y))
