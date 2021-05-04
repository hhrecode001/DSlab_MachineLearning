import numpy as np
from collections import defaultdict


def load_data(data_path):
        def sparse_to_dense(sparse_r_d, vocab_size): # return tf-idf array 
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidfs in indices_tfidfs:
                index = int(index_tfidfs.split(':')[0])
                tfidf = float(index_tfidfs.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines() # read from tf-idf file
        with open('../datasets/20news-bydate/words_idfs.txt') as f:
            vocab_size = len(f.read().splitlines()) # read from words file

        pr = 0
        data = []
        labels = []
        for  d in d_lines: 
            features = d.split('<fff>')
            labels.append(int(features[0]))
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            data.append(r_d)
            if (pr == 0):
                print(r_d[12912])
                pr = 1
        print("Finished load data")
        return np.array(data), labels

def compute_accuracy(predicted_y, expected_y):
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / len(expected_y)
    return accuracy

def clustering_with_KMeans():
    data, labels = load_data(data_path='../datasets/20news-bydate/20news-full-tfidf.txt')

    from sklearn.cluster import KMeans
    from scipy.sparse import  csr_matrix
    X = csr_matrix(data)
    print ('=============')
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2021
    ).fit(X)
    cnt = [[ 0 for _ in range(20)] for _ in range(20)]
    for i in range(len(labels)):
        cnt[ kmeans.labels_[i] ][ labels[i] ] += 1
    for i in range(20):
        print(np.max(cnt[i]), "/", np.sum(cnt[i]))
    

def classifying_with_linear_SVMs():
    train_X, train_Y = load_data(data_path='../datasets/20news-bydate/20news-train-tfidf.txt')
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
        C=10.0,
        tol=0.001,
        verbose=True
    )
    classifier.fit(train_X, train_Y)

    test_X, test_Y = load_data(data_path='../datasets/20news-bydate/20news-test-tfidf.txt')
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_Y, test_Y)
    print("Accuracy: ", accuracy)

def classifying_with_kernel_SVMs():
    train_X, train_Y = load_data(data_path='../datasets/20news-bydate/20news-train-tfidf.txt')
    from sklearn.svm import SVC
    classifier = SVC(
        C=50.0,
        kernel='rbf',
        gamma=0.1,
        tol=0.001,
        verbose=True
    )
    classifier.fit(train_X, train_Y)

    test_X, test_Y = load_data(data_path='../datasets/20news-bydate/20news-test-tfidf.txt')
    predicted_Y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_Y, test_Y)
    print("Accuracy: ", accuracy)

clustering_with_KMeans()
#classifying_with_linear_SVMs()
#classifying_with_kernel_SVMs()
