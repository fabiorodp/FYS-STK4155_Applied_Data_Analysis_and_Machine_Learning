# FYS-STK4155 - H2020 - UiO
# Project 1 - part a
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from package.Create_data import CreateData
from package.Pre_processing import PreProcessing
from package.LinearRegression import LinearRegressionTechniques
from package.Pipeline import MakePipeline
from sklearn.metrics import r2_score, mean_squared_error

# initializing a pipeline
pipe = MakePipeline(
    create_data_model=CreateData(),
    pre_processing_model=PreProcessing(),
    ml_model=LinearRegressionTechniques(technique_name="OLS")
)

for degree in [2, 3, 4, 5]:

    # setting parameters to create the data
    pipe.set_data_model_new_parameters(
        model_name="FrankeFunction",
        nr_samples=100,
        degree=degree,
        seed=10
    )

    # setting pre-processing parameters
    pipe.set_pre_processing_new_parameters(
        test_size=0.25,
        seed=10,
        split=True,
        scale=True
    )

    # fitting and predicting values for [2, 3, 4, 5] degrees
    z_predict_train, z_predict_test = pipe.fit_predict()

    # print coefficients beta
    print("Betas for polynomial of {} degree: ".format(degree),
          pipe.get_coeffs(), "\n")

    # print confidence interval for betas
    print("CI for the betas: ", pipe.get_CI_coeffs_(), "\n")

    # getting necessary parameters
    z_train, z_test = pipe.get_z_values()

    # evaluating training MSE and R2-score
    print("Training MSE : ",
          mean_squared_error(z_train, z_predict_train))
    print("Training R2-score : ",
          r2_score(z_train, z_predict_train), "\n")

    # evaluating test MSE and R2-score
    print("Test MSE : ",
          mean_squared_error(z_train, z_predict_train))
    print("Test R2-score : ",
          r2_score(z_train, z_predict_train), "\n")
