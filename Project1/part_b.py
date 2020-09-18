# FYS-STK4155 - H2020 - UiO
# Project 1 - part b
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from package.Create_data import CreateData
from package.Pre_processing import PreProcessing
from package.LinearRegression import LinearRegressionTechniques
from package.Pipeline import MakePipeline
from package.Metrics import BiasVarianceStudy

# initializing a pipeline
pipe = MakePipeline(
    create_data_model=CreateData(model_name="FrankeFunction", seed=10),
    pre_processing_model=PreProcessing(),
    ml_model=LinearRegressionTechniques(technique_name="OLS")
)

# setting pre-processing parameters
pipe.set_pre_processing_new_parameters(
    test_size=0.25,
    seed=10,
    split=True,
    scale=True
)

# initializing a BiasVarianceStudy
bvs = BiasVarianceStudy(pipe)

# fitting everything
bvs.fit(nr_samples=[100],
        poly_degrees=[2, 3, 4, 5])

# plotting bias-variance trade-off with the predicting errors
# as a function of the complexities (polynomial degrees),
# for a sample space of 100 observations.
bvs.plot_study(function_of="poly_degrees")

# fitting again, but now with degree 5,
# and different number for the sample space
bvs.fit(nr_samples=[10, 50, 100, 200],
        poly_degrees=[5])

# plotting bias-variance trade-off with the predicting errors
# as a function of the complexities (number for the sample space),
# for a polynomial of degree 5.
bvs.plot_study(function_of="nr_samples")
