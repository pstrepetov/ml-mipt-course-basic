import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return : float
            single number with MSE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.
        
        Comment: If Y is two-dimentional, average the error over both dimentions.
        """

        return np.mean((X.dot(w) - Y)**2)

    @staticmethod
    def mae(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
                
        Return: float
            single number with MAE value of linear model (X.dot(w)) with no bias term
            on the selected dataset.
        Comment: If Y is two-dimentional, average the error over both dimentions.
        """

        # YOUR CODE HERE    
        return np.mean(abs(X.dot(w) - Y))

    @staticmethod
    def l2_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return: float
            single number with sum of squared elements of the weight matrix ( \sum_{ij} w_{ij}^2 )
        Computes the L2 regularization term for the weight matrix w.
        """
        
        # YOUR CODE HERE
        w1 = w.reshape((w.size,))
        return (np.linalg.norm(w1, ord=2))**2

    @staticmethod
    def l1_reg(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`)
        Return : float
            single number with sum of the absolute values of the weight matrix ( \sum_{ij} |w_{ij}| )
        
        Computes the L1 regularization term for the weight matrix w.
        """

        # YOUR CODE HERE
        w1 = w.reshape((w.size,))
        return np.linalg.norm(w, ord=1)

    @staticmethod
    def no_reg(w):
        """
        Simply ignores the regularization
        """
        return 0.
    
    @staticmethod
    def mse_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        
        Return : numpy array of same shape as `w`
        Computes the MSE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """

        # YOUR CODE HERE
        derivative = 2*X.T.dot(X.dot(w)-Y)
        return derivative/Y.size

    @staticmethod
    def mae_derivative(X, Y, w):
        """
        X : numpy array of shape (`n_observations`, `n_features`)
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        
        Return : numpy array of same shape as `w`
        Computes the MAE derivative for linear regression (X.dot(w)) with no bias term
        w.r.t. w weight matrix.
        
        Please mention, that in case `target_dimentionality` > 1 the error is averaged along this
        dimension as well, so you need to consider that fact in derivative implementation.
        """

        # YOUR CODE HERE
        n_observations = int(X.shape[0])
        n_features = int(X.shape[1])
        target_dim = int(Y.size/n_observations)
        
        answer = np.zeros((target_dim, n_features))
        #max_ind = []
        #sum_col = np.zeros(target_dim)
        #for j in range(target_dim):
        #    summ = 0
        #    for i in range(n_observations):
        #        summ += abs((X.dot(w) - Y)[i][j])
        #    sum_col[j] = summ
        #args = np.argsort(sum_col)
        #end = target_dim - 1
        #maxi = sum_col[args[end]]
        #while (end >= 0 and maxi == sum_col[args[end]]):
        #    max_ind.append(args[end])
        #    end -= 1
        for j in range(target_dim):
            for i in range(n_observations):
                answer[j] += X[i]*np.sign((X.dot(w) - Y)[i][j])
        #return answer.T/(Y.size*len(max_ind))
        return answer.T/(Y.size)

    @staticmethod
    def l2_reg_derivative(w):
        """
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return : numpy array of same shape as `w`
        Computes the L2 regularization term derivative w.r.t. the weight matrix w.
        """

        # YOUR CODE HERE
        return 2*w

    @staticmethod
    def l1_reg_derivative(w):
        """
        Y : numpy array of shape (`n_observations`, `target_dimentionality`) or (`n_observations`,)
        w : numpy array of shape (`n_features`, `target_dimentionality`) or (`n_features`,)
        Return : numpy array of same shape as `w`
        Computes the L1 regularization term derivative w.r.t. the weight matrix w.
        """

        # YOUR CODE HERE
        n_features = int(w.shape[0])
        target_dim = int(w.size/n_features)
        X = np.eye(n_features)
        Y = np.zeros((n_features, target_dim))
        answer = np.zeros((target_dim, n_features))
        for j in range(target_dim):
            for i in range(n_features):
                answer[j] += X[i]*np.sign((X.dot(w) - Y)[i][j])
        return answer.T
        #return mae_derivative(X, Y, w)

    @staticmethod
    def no_reg_derivative(w):
        """
        Simply ignores the derivative
        """
        return np.zeros_like(w)
