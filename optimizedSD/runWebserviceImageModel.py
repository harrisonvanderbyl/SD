from io import BytesIO
from itertools import islice
import time
from pytorch_lightning import seed_everything
import base64
import urllib3, json 
from modelArguments import ModelArgValues, ModelArguments 
from model import Model
import os
http = urllib3.PoolManager()

DEFAULT_HEADERS = {'content-type': 'text/plain'}
REQUEST_FAILED = "Request failed"
def makeHttpRequest(type, url, body = None):
    try:
        if type == "GET":
            return http.request("GET", url)
        elif type == "POST":
            return http.request("POST", url, body=body, headers=DEFAULT_HEADERS)
    except Exception as e:
        print(e)
        return REQUEST_FAILED

initialModelOptions = ModelArguments.parseFromConsoleArguments()

seed_everything(initialModelOptions.seed)
models = Model(initialModelOptions)

DEFAULT_SERVER_REQUEST_DELAY = 10

def parseModelOptionsFromServerResponse(serverResponse: dict) -> ModelArguments:
    prompt = serverResponse["prompt"]
    assert prompt is not None
    
    seed = serverResponse["seed"]
    if(float(seed)<1.0):
        seed = seed.replace(".","")
    seed = int(seed)    
    
    ddim_steps = int(serverResponse["samples"])

    
    return ModelArguments(
        ModelArgValues(
        prompt = prompt,
        outdir = initialModelOptions.outdir,
        skip_grid = initialModelOptions.skip_grid,
        skip_save = initialModelOptions.skip_save,
        ddim_steps = ddim_steps,
        fixed_code = initialModelOptions.fixed_code,
        ddim_eta = initialModelOptions.ddim_eta,
        n_iter = initialModelOptions.n_iter,
        H = initialModelOptions.H,
        W = initialModelOptions.W,
        C = initialModelOptions.C,
        f = initialModelOptions.f,
        n_samples = initialModelOptions.n_samples,
        n_rows = initialModelOptions.n_rows,
        scale = initialModelOptions.scale,
        from_file = initialModelOptions.from_file,
        seed = seed,
        small_batch = initialModelOptions.small_batch,
        precision = initialModelOptions.precision,
        url = initialModelOptions.url,
        config = initialModelOptions.config,
        ckpt = initialModelOptions.ckpt,
        device = initialModelOptions.device,
        inputimg = initialModelOptions.inputimg,
        
    ))
print("model loaded")
while True:
    print("atarting listen")
    fetchedNewPromptFromServer = False
    while not fetchedNewPromptFromServer:
        serverData = makeHttpRequest("GET",f"{initialModelOptions.url}/sdlist")
        if serverData == REQUEST_FAILED:
            time.sleep(DEFAULT_SERVER_REQUEST_DELAY)
            continue

        serverResponse = json.loads(serverData.data.decode("utf-8"))
        print(serverResponse)
        if(not "prompt" in serverResponse):
            print("No prompt specified. Waiting before making a new request")
            time.sleep(DEFAULT_SERVER_REQUEST_DELAY)
        else:
            print(f"Using prompt:{serverResponse['prompt']}")
            fetchedNewPromptFromServer = True

    pid = serverResponse["id"]
    modelOptions = parseModelOptionsFromServerResponse(serverResponse)
    inputimg = None
    if "input" in serverResponse:
        print("inputimg found")
        inputimg = serverResponse["input"]
    strength = "0.5"
    if "strength" in serverResponse:
        print("strength found")
        strength = serverResponse["strength"]
    models.config.modelUNet.params.ddim_steps = modelOptions.ddim_steps
    seed_everything(modelOptions.seed)

    makeHttpRequest("POST", f"{modelOptions.url}/update/{pid}", body=f"starting with seed {modelOptions.seed}")
    data = [modelOptions.n_samples * [modelOptions.prompt]]

    def updateText(i):
        makeHttpRequest("POST", f"{modelOptions.url}/update/{pid}", body=f"Progress: {i}/{modelOptions.ddim_steps}")

    def saveCallback(image):
        temp = BytesIO()
        image.save(temp,"jpeg")
        encoded = base64.b64encode(temp.getvalue()).decode('utf-8')
        saveFailed = makeHttpRequest("POST", f"{modelOptions.url}/upload/{pid}", body=encoded)
        if saveFailed == REQUEST_FAILED:
            imageCount = len(os.listdir("./outputs/fallbackForWebserver/"))
            image.save(f"./outputs/fallbackForWebserver/image_{imageCount}.jpeg","jpeg")

    models.sampleFromModel(modelOptions, data, updateText, saveCallback,inputimg,strength)
    "Finish generating images"