# FYS-STK4155 - H2020 - UiO
# Project 1 - part b
# Author: Fábio Rodrigues Pereira
# E-mail: fabior@uio.no


from package.Create_data import CreateData
from package.Pre_processing import PreProcessing
from package.LinearRegression import OlsModel
from package.Pipeline import MakePipeline
from package.Metrics import ComplexityStudy, Bootstrap


# initializing a pipeline
pipe = MakePipeline(
    create_data_model=CreateData(model_name="FrankeFunction",
                                 seed=10),
    pre_processing_model=PreProcessing(),
    ml_model=OlsModel()
)

# setting pre-processing parameters
pipe.set_Preprocessing_parameters(
    test_size=0.25,
    seed=10,
    split=True,
    scale=True
)

# initializing a ComplexityStudy
cs = ComplexityStudy(pipe)

# fitting everything
cs.fit(nr_samples=[100],
       poly_degrees=[2, 3, 4, 5])

# plotting ComplexityStudy with the predicting errors
# as a function of the complexities (polynomial degrees),
# for a sample space of 100 observations.
cs.plot_study(function_of="poly_degrees")

# fitting again, but now with degree 5,
# and different numbers for the sample space
cs.fit(nr_samples=[10, 50, 100, 200],
       poly_degrees=[5])

# plotting ComplexityStudy with the predicting errors
# as a function of the complexities (size for the sample space),
# for a polynomial of degree 5.
cs.plot_study(function_of="nr_samples")

# 2nd part of the exercise
bp = Bootstrap(CreateData, PreProcessing, OlsModel, seed=10)
bp.plot(n_boostraps=100, maxdegree=20, verbose=True)
