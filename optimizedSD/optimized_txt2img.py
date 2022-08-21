from io import BytesIO
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import base64
import urllib3, json 
from modelArguments import ModelArguments 
from model import Model
http = urllib3.PoolManager()


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

modelOptions = ModelArguments.parseFromConsoleArguments()

seed_everything(modelOptions.seed)
models = Model(modelOptions)

while True:
    found = False
    prdata = {}
    while not found:
        sdlist = http.request("GET",f"{modelOptions.url}/sdlist") 
        prdata = json.loads(sdlist.data.decode("utf-8"))
        print(prdata)
        print(sdlist.data.decode("utf-8"))
        if(not "prompt" in prdata):
            print("No prompt specified. Using default prompt: a painting of a virus monster playing guitar")
            #wait 1 second to avoid overloading the server
            time.sleep(1)
        else:
            print(f"Using prompt:"+prdata["prompt"])
            found = True


    modelOptions.prompt = prdata["prompt"]
    pid = prdata["id"]
    if(float(prdata["seed"])<1.0):
        prdata["seed"] = prdata["seed"].replace(".","")
        
    modelOptions.ddim_steps = int(prdata["samples"])
    models.config.modelUNet.params.ddim_steps = modelOptions.ddim_steps

    seed_everything(int(prdata["seed"]))

    def updateText(i):
        req = http.request("POST", f"{modelOptions.url}/update/{pid}", body="seed:"+ prdata["seed"]+"\nProgress:"+str(i)+"/50", headers=headers)
   

    start_code = torch.randn([modelOptions.n_samples, modelOptions.C, modelOptions.H // modelOptions.f, modelOptions.W // modelOptions.f], device=modelOptions.device)

    headers = {'content-type': 'text/plain'}
    req = http.request("POST", f"{modelOptions.url}/update/{pid}", body="starting with seed "+ prdata["seed"], headers=headers)
    if not modelOptions.from_file:
        prompt = modelOptions.prompt
        assert prompt is not None
        data = [modelOptions.n_samples * [prompt]]

    else:
        print(f"reading prompts from {modelOptions.from_file}")
        with open(modelOptions.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, modelOptions.n_samples))

    precision_scope = autocast if modelOptions.precision=="autocast" else nullcontext
    with torch.no_grad():

        all_samples = list()
        # download prompt
        
        for n in trange(modelOptions.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    models.modelCS.to(modelOptions.device)
                    uc = None
                    if modelOptions.scale != 1.0:
                        uc = models.modelCS.get_learned_conditioning(modelOptions.n_samples * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    
                    c = models.modelCS.get_learned_conditioning(prompts)
                    shape = [modelOptions.C, modelOptions.H // modelOptions.f, modelOptions.W // modelOptions.f]
                    mem = torch.cuda.memory_allocated()/1e6
                    models.modelCS.to("cpu")
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)


                    samples_ddim = models.model.sample(S=modelOptions.ddim_steps,
                                    conditioning=c,
                                    batch_size=modelOptions.n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=modelOptions.scale,
                                    unconditional_conditioning=uc,
                                    eta=modelOptions.ddim_eta,
                                    x_T=start_code,
                                    callback=updateText
                                    )

                    models.modelFS.to(modelOptions.device)
                    print("saving images")
                    for i in range(modelOptions.n_samples):
                        
                        x_samples_ddim = models.modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    # for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        # send image using post request as base64
                        temp = BytesIO()
                        
                        image.save(temp,"jpeg")
                        encoded = base64.b64encode(temp.getvalue()).decode('utf-8')
                    
                        headers = {'content-type': 'text/plain'}
                        req = http.request("POST", f"{modelOptions.url}/upload/{pid}", body=encoded, headers=headers)


                    mem = torch.cuda.memory_allocated()/1e6
                    models.modelFS.to("cpu")
                    while(torch.cuda.memory_allocated()/1e6 >= mem):
                        time.sleep(1)

                    # if not opt.skip_grid:
                    #     all_samples.append(x_samples_ddim)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated()/1e6)