import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from modelArguments import ModelArguments 

def load_state_dictionary_from_config(ckptFilePath):
    print(f"Loading model from {ckptFilePath}")
    pl_sd = torch.load(ckptFilePath, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    stateDictionary = pl_sd["state_dict"]
    return stateDictionary

class Model:
    def __init__(self, modelOptions: ModelArguments):
        stateDictionary = load_state_dictionary_from_config(modelOptions.ckpt)
        li = []
        lo = []
        for key, value in stateDictionary.items():
            sp = key.split('.')
            if(sp[0]) == 'model':
                if('input_blocks' in sp):
                    li.append(key)
                elif('middle_block' in sp):
                    li.append(key)
                elif('time_embed' in sp):
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            stateDictionary['model1.' + key[6:]] = stateDictionary.pop(key)
        for key in lo:
            stateDictionary['model2.' + key[6:]] = stateDictionary.pop(key)

        self.config = OmegaConf.load(f"{modelOptions.config}")

        self.config.modelUNet.params.small_batch = modelOptions.small_batch

        self.model = instantiate_from_config(self.config.modelUNet)
        self.model.load_state_dict(stateDictionary, strict=False)
        self.model.eval()
            
        self.modelCS = instantiate_from_config(self.config.modelCondStage)
        self.modelCS.load_state_dict(stateDictionary, strict=False)
        self.modelCS.eval()
            
        self.modelFS = instantiate_from_config(self.config.modelFirstStage)
        self.modelFS.load_state_dict(stateDictionary, strict=False)
        self.modelFS.eval()

        if modelOptions.precision == "autocast":
            self.model.half()
            self.modelCS.half()
