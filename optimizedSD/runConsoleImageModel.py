from itertools import islice
from pytorch_lightning import seed_everything
from modelArguments import ModelArguments 
from model import Model
import os

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

modelOptions = ModelArguments.parseFromConsoleArguments()

seed_everything(modelOptions.seed)
models = Model(modelOptions)
    

models.config.modelUNet.params.ddim_steps = modelOptions.ddim_steps
seed_everything(modelOptions.seed)

data = [modelOptions.n_samples * [modelOptions.prompt]]

def saveCallback(image):
    imageCount = len(os.listdir("./outputs/fallbackForWebserver/"))
    image.save(f"./outputs/fallbackForWebserver/image_{imageCount}.jpeg","jpeg")

models.sampleFromModel(modelOptions, data, lambda _: None, saveCallback)