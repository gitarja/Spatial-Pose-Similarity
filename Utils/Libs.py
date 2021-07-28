from sklearn.neighbors import KNeighborsClassifier

def kNN(query, q_labels, references, ref_labels, ref_num=1, th=0.5):
    X = references.numpy()
    y = ref_labels + 1
    classifier = KNeighborsClassifier(n_neighbors=ref_num, metric="euclidean", algorithm="kd_tree")
    classifier.fit(X, y)

    q = query.numpy()
    q_y = q_labels + 1

    return classifier.predict(q) == q_y




