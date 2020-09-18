# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PreProcessing():
    def __init__(self, test_size=None, seed=None, split=True, scale=True):
        self.test_size = test_size
        self.seed = seed
        self.split = split
        self.scale = scale
        self.X_train = None
        self.X_test = None
        self.z_train = None
        self.z_test = None

    def fit(self, X, z):
        if (self.test_size is None) or (self.seed is None):
            raise ValueError("Error: It can not fit if there are any "
                             "'None' parameter. Please set new parameters "
                             "before fit.")

        if (self.split is True) and (self.scale is True):
            self.split_data(X, z)
            self.scale_data()

    def split_data(self, X, z):
        # splitting the data in test and training data
        self.X_train, self.X_test, self.z_train, self.z_test = \
            train_test_split(X, z, test_size=self.test_size,
                             random_state=self.seed)

    def scale_data(self):
        # scaling X_train and X_test
        scaler = StandardScaler().fit(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        self.X_train = scaler.transform(self.X_train)

    def bootstrap(self):
        # t = zeros(R);
        # n = len(data);
        # inds = arange(n);
        # t0 = time()
        # # non-parametric bootstrap
        # for i in range(R):
        #     t[i] = statistic(data[randint(0, n, n)])
        #
        # # analysis
        # print("Runtime: %g sec" % (time() - t0));
        # print("Bootstrap Statistics :")
        # print("original           bias      std. error")
        # print("%8g %8g %14g %15g" % (statistic(data), std(data), mean(t), std(t)))
        # return t
        pass

    def cv_k_fold(self):
        pass

    def set_new_parameters(self, test_size, seed, split=True, scale=True):
        self.test_size = test_size
        self.seed = seed
        self.split = split
        self.scale = scale
