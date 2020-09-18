# FYS-STK4155 - H2020 - UiO
# Project 1
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


class MakePipeline:

    def __init__(self, create_data_model, pre_processing_model, ml_model):
        self.data_model = create_data_model
        self.pre_processing_model = pre_processing_model
        self.ml_model = ml_model

    def fit_predict(self):
        self.data_model.fit()

        self.pre_processing_model.fit(
            self.data_model.X,
            self.data_model.z)

        self.ml_model.fit(
            self.pre_processing_model.X_train,
            self.pre_processing_model.z_train)

        z_predict_train = \
            self.ml_model.predict(self.pre_processing_model.X_train)

        z_predict_test = \
            self.ml_model.predict(self.pre_processing_model.X_test)

        return z_predict_train, z_predict_test

    def get_X_values(self):
        X_train = self.pre_processing_model.X_train
        X_test = self.pre_processing_model.X_test
        return X_train, X_test

    def get_z_values(self):
        z_train = self.pre_processing_model.z_train
        z_test = self.pre_processing_model.z_test
        return z_train, z_test

    def get_coeffs(self):
        betas = self.ml_model.coef_OLS
        return betas

    def get_CI_coeffs_(self, percentile=0.95):
        CI = self.ml_model.beta_CI_OLS(percentile)
        return CI

    def get_data_model_parameters(self):
        return self.data_model.model, self.data_model.nr_samples, \
               self.data_model.degree, self.data_model.seed

    def set_data_model_new_parameters(
            self, model_name, nr_samples, degree, seed):
        self.data_model.set_new_parameters(
            model_name, nr_samples, degree, seed)

    def set_pre_processing_new_parameters(
            self, test_size, seed, split=True, scale=True):
        self.pre_processing_model.set_new_parameters(
            test_size, seed, split, scale)

    def set_ml_model_new_parameters(self, technique_name="OLS"):
        self.ml_model.set_new_parameters(technique_name)
