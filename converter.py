import sys
import coremltools
from keras.models import load_model

originalFile = sys.argv[1]
convertedFile = sys.argv[2]

angleModel = coremltools.converters.keras.convert(str(originalFile), input_names=["pitch"], output_names=["widthHeightRatio"])

angleModel.author = 'Weisu Yin'
angleModel.short_description = 'Pitch angle regression'
angleModel.input_description['pitch'] = 'Takes as input a pitch angle of a mobile device'
angleModel.output_description['widthHeightRatio'] = 'Prediction of widthHeightRatio of a credit card'

angleModel.save(convertedFile)
