from itertools import islice
from pytorch_lightning import seed_everything
from modelArguments import ModelArguments 
from model import Model
import os

modelOptions = ModelArguments.parseFromConsoleArguments()

seed_everything(modelOptions.seed)
models = Model(modelOptions)

data = [modelOptions.n_samples * [modelOptions.prompt]]

def saveCallback(image):
    imageCount = len(os.listdir("./outputs/"))
    image.save(f"./outputs/image_{imageCount}.jpeg","jpeg")

models.sampleFromModel(modelOptions, data, lambda _: None, saveCallback)