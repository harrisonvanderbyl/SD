from io import BytesIO
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from modelArguments import ModelArguments 
import time
from tqdm import tqdm, trange
from einops import rearrange
from PIL import Image
import numpy as np
import base64
def load_img(base64text, h0, w0):
    temp = BytesIO()
    
    temp.write(base64.b64decode(base64text))
    
    image = Image.open(temp,"r",["jpeg"]).convert("RGB")
    w, h = image.size
   
    if(h0 is not None and w0 is not None):
        h, w = h0, w0
    
    w, h = map(lambda x: x - x % 32, (w0, h0))  # resize to integer multiple of 32

    print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample = Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

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
        self.config.modelUNet.params.ddim_steps = modelOptions.ddim_steps

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
            self.modelFS.half()
    
    def sampleFromModel(self, modelOptions, data, updateCallback, saveCallback, inputimg=None, inputimgstrength=None):
        if(inputimg is not None):
            print(f"Loading image from {len(inputimg)} chars and {len(inputimgstrength)} strength")
            self.modelFS.to(modelOptions.device)
            init_image = load_img(inputimg, modelOptions.H, modelOptions.W).to("cuda")
            if modelOptions.precision == "autocast":
                init_image = init_image.half()
            init_latent = self.modelFS.get_first_stage_encoding(self.modelFS.encode_first_stage(init_image))  # move to latent space
            self.modelFS.to("cpu")
            if(inputimgstrength is not None):
                t_enc = int(float(inputimgstrength) * 50)
            else:
                t_enc = int(0.9 * modelOptions.ddim_steps)
        else:
            start_code = torch.randn(modelOptions.getTorchShape(), device=modelOptions.device)
        with torch.no_grad():
            for n in trange(modelOptions.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    with modelOptions.precision_scope("cuda"):
                        self.modelCS.to(modelOptions.device)
                        uc = None
                        if modelOptions.scale != 1.0:
                            uc = self.modelCS.get_learned_conditioning(modelOptions.n_samples * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c = self.modelCS.get_learned_conditioning(prompts)
                        shape = modelOptions.getShape()
                        mem = torch.cuda.memory_allocated()/1e6
                        self.modelCS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)

                        if(inputimg is None):
                            samples_ddim = self.model.sample(S=modelOptions.ddim_steps,
                                        conditioning=c,
                                        batch_size=modelOptions.n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=modelOptions.scale,
                                        unconditional_conditioning=uc,
                                        eta=modelOptions.ddim_eta,
                                        x_T=start_code,
                                        callback=updateCallback
                                        )
                        # encode (scaled latent)
                        else:
                            z_enc = self.model.stochastic_encode(init_latent, torch.tensor([t_enc]*1).to(modelOptions.device))
                        # decode it
                            samples_ddim = self.model.decode(z_enc, c, t_enc, unconditional_guidance_scale=modelOptions.scale,
                                                    unconditional_conditioning=uc,)
                        self.modelFS.to(modelOptions.device)
                        print("saving images")
                        for i in range(modelOptions.n_samples):
                            x_samples_ddim = self.modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                            image = Image.fromarray(x_sample.astype(np.uint8))
                            saveCallback(image)

                        mem = torch.cuda.memory_allocated()/1e6
                        self.modelFS.to("cpu")
                        while(torch.cuda.memory_allocated()/1e6 >= mem):
                            time.sleep(1)

                        del samples_ddim
                        print("memory_final = ", torch.cuda.memory_allocated()/1e6)
