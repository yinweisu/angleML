import coremltools
from keras.models import load_model

angleModel = coremltools.converters.keras.convert("angleModel.h5", input_names=["widthHeightRatio"], output_names=["pitch"])

angleModel.author = 'Weisu Yin'
angleModel.short_description = 'Pitch angle regression'
angleModel.input_description['ratio'] = 'Takes as input a widthHeightRatio of a credit card'
angleModel.output_description['output'] = 'Prediction of Pitch Angle'

angleModel.save("angleModel.mlmodel")
