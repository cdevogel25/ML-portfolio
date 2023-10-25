import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
dataset = loadmat('face_emotion_data.mat')

X, y = dataset['X'], dataset['y']
n, p = np.shape(X)

### don't do this, if you do threshold at 1/2
# y[y==-1] = 0
X = np.hstack((np.ones((n,1)), X)) # append a column of ones

def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1 - x2)**2/(2*sigma**2))

class KernelClassifier:
    def __init__(self, kernel, lmb=0.5):
        self.kernel = kernel
        self.lmb = lmb
        self.alpha = None

    def fit(self, X, y, sigma):
        K = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                # set the kernel-output matrix values for each pair of points
                K[i,j] = self.kernel(X[i,:], X[j,:], sigma)
        
        # find the resulting alpha vector for the given sigma/lambda values
        self.alpha = np.linalg.inv(K + self.lmb * np.identity(K.shape[0]))@y

    def predict(self, X_test, X_train, y_train, sigma):
        K_t = np.zeros((X_test.shape[0], X_train.shape[0]))

        for i in range(K_t.shape[0]):
            for j in range(K_t.shape[1]):
                K_t[i,j] = self.kernel(X_test[i,:], X_train[j,:], sigma)
        
        ### do below if using a -1/+1 label
        y_hat = np.sign(K_t@self.alpha)

        ### do below if using a 1/0 label
        # y_hat = K_t@self.alpha
        # y_hat[yhat == 0.5] = 0

        return y_hat

# 8-fold cross-validator
class CrossValidator8f:
    def __init__(self, classifier, X, y, min_sigma, max_sigma, npoints, shuffle=True):
        self.classifier = classifier
        self.sigma_range = np.linspace(min_sigma, max_sigma, npoints)
        self.npoints = npoints

        if(shuffle):
            shuffle_data = np.hstack((X,y))
            np.random.shuffle(shuffle_data)

            self.X_full = shuffle_data[:,:-1]
            self.y_full = np.vstack(shuffle_data[:,-1])
        else:
            self.X_full = X
            self_y_full = y

        # figure out how to do this programmatically
        set_indices = [[1, 16], [17, 32], [33, 48], [49, 64], [65, 80], [81, 96], [97, 112], [113,128]]
        holdout_indices = [[1], [2], [3], [4], [5], [6], [7], [8]]
        cases = len(holdout_indices)

        err_total = 0.

        for j in range(cases):
            validate_ind = np.arange(set_indices[holdout_indices[j][0] - 1][0] - 1,
                                     set_indices[holdout_indices[j][0] - 1][1])
            
            train_ind = list(set(range(128)) - set(validate_ind))

            A_validate = self.X_full[validate_ind,:]
            b_validate = self.y_full[validate_ind]

            A_train = self.X_full[train_ind,:]
            b_train = self.y_full[train_ind]

            error_rate = np.zeros(self.npoints)
            for i in range(self.npoints):
                self.classifier.fit(A_train, b_train, self.sigma_range[i])
                y_hat = self.classifier.predict(A_train, A_train, b_train, self.sigma_range[i])
                error_rate[i] = np.sum(np.abs(y_hat - b_train)) / b_train.shape[0]
            
            opt_sigma_ind = np.argmin(error_rate)
            opt_sigma = self.sigma_range[opt_sigma_ind]

            self.classifier.fit(A_train, b_train, opt_sigma)

            error = np.sum(np.abs(y_hat - b_validate)) / y_hat.shape[0]
            error_total += error
        
        error_avg = error_total / cases
        print(error_avg)

