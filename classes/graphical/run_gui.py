from ModelSwitcher import ModelSwitcher
from FaceMapper import FaceMapper

MS = ModelSwitcher(
    normalisation_method=FaceMapper.NormalisationMethods.MOUTH_LOCATION_KEY
)
MS.start()
