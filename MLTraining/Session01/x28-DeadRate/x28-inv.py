import numpy as np 

# initiate
MAXF = 15

feature = []
label = []

# input
def get_data(X,Y):
    with open('x28.txt','r') as f:
        content = f.read()
        f.close()

    ## ignore the instruction in .txt file
    key_word = "B death rate"
    if (content.find(key_word) > 0):
        data_begin = content.find(key_word) + len(key_word)
        data_input = content[data_begin:(len(content))]

    ## get the data
    data_input = data_input.split()
    i=0
    while (i < len(data_input)):
        X_row = []
        i += 1
        for _ in range(1,MAXF+1):
            X_row.append(float(data_input[i]))
            i += 1
            
        Y.append(float(data_input[i]))
        i += 1
        X.append(X_row)

# feature scaling
def normallize_and_add_ones(X):
    X = np.array(X)
    X_max = np.array([[np.amax(X[:, column_id]) for column_id in range(X.shape[1])] for _ in range(X.shape[0])])
    X_min = np.array([[np.amin(X[:, column_id]) for column_id in range(X.shape[1])] for _ in range(X.shape[0])])

    X_normalized = (X - X_min) / (X_max - X_min)

    ones = np.array([[1] for _ in range(X_normalized.shape[0])])
    return np.column_stack((ones, X_normalized))

# RidgeRegression Class
class RidgeRegression:
    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMB):
        assert (len(X_train.shape) == 2) and (X_train.shape[0] == Y_train.shape[0])
        W = np.linalg.inv( X_train.transpose().dot(X_train) + LAMB * np.identity(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
        return W

    def predict(self, W, X_new):
        X_new = np.array(X_new)
        Y_new = X_new.dot(W)
        return Y_new

    def compute_RSS(self, Y_new, Y_predicted, Yprint = 0):
        if (Yprint == 1):
            print("Actuality          Prediction")
            for i in range(len(Y_new)):
                print(Y_new[i]," ",Y_predicted[i])
        loss = 1./ Y_new.shape[0] * np.sum((Y_new - Y_predicted) ** 2)
        return loss

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMB):
            row_ids = np.array( range(X_train.shape[0]))
            valid_ids = np.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
            valid_ids[-1] = np.append(valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in valid_ids[i]] for i in range(num_folds)]
            aver_RSS = 0
            for i in range(num_folds):
                valid_part = {'X': X_train[valid_ids[i]], 'Y': Y_train[valid_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['Y'], LAMB)
                Y_predicted = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predicted)
            return aver_RSS / num_folds
        
        def range_scan(best_LAMB, minimum_RSS, LAMB_values):
            for current_LAMB in LAMB_values:
                aver_RSS = cross_validation(num_folds = 5, LAMB = current_LAMB)
                if (aver_RSS < minimum_RSS):
                    best_LAMB = current_LAMB
                    minimum_RSS = aver_RSS
            return best_LAMB, minimum_RSS

        best_LAMB ,  minimum_RSS = range_scan( best_LAMB = 0, minimum_RSS=10000 ** 2, LAMB_values = range(50)) 

        # tiep tuc chia nho LAMB_values de tinh
        LAMB_values = [k*1./ 1000 for k in range( max(0,(best_LAMB - 1) * 1000), (best_LAMB + 1) * 1000 , 1)]
        best_LAMB , minimum_RSS = range_scan(best_LAMB = best_LAMB, minimum_RSS = minimum_RSS, LAMB_values = LAMB_values)

        return best_LAMB

# main 

get_data(feature,label) #doc du lieu

feature = normallize_and_add_ones(feature) # chuan hoa du lieu
label = np.array(label)

# shuffle
#arr = np.array(range(X_train.shape[0]))
#np.random.shuffle(arr)
#feature = feature[arr]
#feature = feature[arr]

feature_train = feature[:50] # chia du lieu 50train-10test
feature_test = feature[50:]
label_train = label[:50]
label_test = label[50:]

ridgeRegression = RidgeRegression()

best_LAMBDA = ridgeRegression.get_the_best_LAMBDA(feature_train,label_train)
W_learned = ridgeRegression.fit(X_train = feature_train, Y_train = label_train, LAMB = best_LAMBDA)
label_predicted = ridgeRegression.predict( W = W_learned, X_new = feature_test)
print ("Loss: ",ridgeRegression.compute_RSS(Y_new = label_test, Y_predicted = label_predicted, Yprint = 1))
print ("Best LAMBDA ", best_LAMBDA)

