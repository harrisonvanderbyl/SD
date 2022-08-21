from dataclasses import dataclass
import argparse
from torch import autocast
from contextlib import nullcontext
import sys

@dataclass
class ParsedConsoleArgValues:
    prompt: str
    outdir: str
    skip_grid: bool
    skip_save: bool
    ddim_steps: int
    fixed_code: bool
    ddim_eta: float
    n_iter: int
    H: int
    W: int
    C: int
    f: int
    n_samples: int
    n_rows: int
    scale: float
    from_file: str
    seed: int
    small_batch: bool
    precision: str
    url: str

@dataclass
class ModelArgValues(ParsedConsoleArgValues):
    config: str
    ckpt: str
    device: str

defaultModelArgs = ModelArgValues(
    prompt="a painting of a virus monster playing guitar",
    outdir="outputs/txt2img-samples",
    skip_grid= False,
    skip_save=False,
    ddim_steps=20,
    fixed_code=False,
    ddim_eta=0.0,
    n_iter=1,
    H=512,
    W=512,
    C=4,
    f=8,
    n_samples=1,
    n_rows=0,
    scale=7.5,
    from_file=None,
    seed=42,
    small_batch=False,
    precision="autocast",
    url="http://localhost:8080",
    config = "optimizedSD/v1-inference.yaml",
    ckpt = "models/ldm/stable-diffusion-v1/model.ckpt",
    device = "cuda"
)

class ModelArguments:
    defaultModelArgs = defaultModelArgs

    @classmethod
    def setDefaultValues(cls, modelArguments: ModelArgValues):
        cls.defaultModelArgs = modelArguments

    def __init__(self, modelArguments: ModelArgValues):
        # Probably can do this better, whatever
        self.prompt = modelArguments.prompt
        self.outdir = modelArguments.outdir
        self.skip_grid = modelArguments.skip_grid
        self.skip_save = modelArguments.skip_save
        self.ddim_steps = modelArguments.ddim_steps
        self.fixed_code = modelArguments.fixed_code
        self.ddim_eta = modelArguments.ddim_eta
        self.n_iter = modelArguments.n_iter
        self.H = modelArguments.H
        self.W = modelArguments.W
        self.C = modelArguments.C
        self.f = modelArguments.f
        self.n_samples = modelArguments.n_samples
        self.n_rows = modelArguments.n_rows
        self.scale = modelArguments.scale
        self.from_file = modelArguments.from_file
        self.seed = modelArguments.seed
        self.small_batch = modelArguments.small_batch
        self.precision = modelArguments.precision
        self.url = modelArguments.url
        self.config = modelArguments.config
        self.ckpt = modelArguments.ckpt
        self.device = modelArguments.device

    def getShape(self):
        return [self.C, self.H // self.f, self.W // self.f]

    def getTorchShape(self):
        return [self.n_samples, *self.getShape()]

    @property
    def precision_scope(self):
        return autocast if self.precision=="autocast" else nullcontext

    @classmethod
    def parseFromConsoleArguments(cls):
        args = sys.argv
        args.pop(0)
        return cls.parseArguments(args)

    @classmethod
    def parseArguments(cls, argList):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--prompt",
            type=str,
            nargs="?",
            default=cls.defaultModelArgs.prompt,
            help="the prompt to render"
        )
        parser.add_argument(
            "--outdir",
            type=str,
            nargs="?",
            default=cls.defaultModelArgs.outdir,
            help="dir to write results to",
        )
        parser.add_argument(
            "--skip_grid",
            type=bool,
            default=cls.defaultModelArgs.skip_grid,
            help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
        )
        parser.add_argument(
            "--skip_save",
            type=bool,
            default=cls.defaultModelArgs.skip_save,
            help="do not save individual samples. For speed measurements.",
        )
        parser.add_argument(
            "--ddim_steps",
            type=int,
            default=cls.defaultModelArgs.ddim_steps,
            help="number of ddim sampling steps",
        )
        parser.add_argument(
            "--fixed_code",
            type=bool,
            default=cls.defaultModelArgs.fixed_code,
            help="if enabled, uses the same starting code across samples ",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=cls.defaultModelArgs.ddim_eta,
            help="ddim eta (eta=0.0 corresponds to deterministic sampling",
        )
        parser.add_argument(
            "--n_iter",
            type=int,
            default=cls.defaultModelArgs.n_iter,
            help="sample this often",
        )
        parser.add_argument(
            "--H",
            type=int,
            default=cls.defaultModelArgs.H,
            help="image height, in pixel space",
        )
        parser.add_argument(
            "--W",
            type=int,
            default=cls.defaultModelArgs.W,
            help="image width, in pixel space",
        )
        parser.add_argument(
            "--C",
            type=int,
            default=cls.defaultModelArgs.C,
            help="latent channels",
        )
        parser.add_argument(
            "--f",
            type=int,
            default=cls.defaultModelArgs.f,
            help="downsampling factor",
        )
        parser.add_argument(
            "--n_samples",
            type=int,
            default=cls.defaultModelArgs.n_samples,
            help="how many samples to produce for each given prompt. A.k.a. batch size",
        )
        parser.add_argument(
            "--n_rows",
            type=int,
            default=cls.defaultModelArgs.n_rows,
            help="rows in the grid (default: n_samples)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=cls.defaultModelArgs.scale,
            help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
        )
        parser.add_argument(
            "--from_file",
            type=str,
            default=cls.defaultModelArgs.from_file,
            help="if specified, load prompts from this file",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=cls.defaultModelArgs.seed,
            help="the seed (for reproducible sampling)",
        )
        parser.add_argument(
            "--small_batch",
            type=bool,
            default=cls.defaultModelArgs.small_batch,
            help="Reduce inference time when generate a smaller batch of images",
        )
        parser.add_argument(
            "--precision",
            type=str,
            help="evaluate at this precision",
            choices=["full", "autocast"],
            default=cls.defaultModelArgs.precision,
        )
        parser.add_argument(
            "--url",
            type=str,
            help="url of associated writerbot instance",
            default=cls.defaultModelArgs.url,
        )
        opt = parser.parse_args(argList)
        return cls(ModelArgValues(
                    prompt = opt.prompt,
                    outdir = opt.outdir,
                    skip_grid = opt.skip_grid,
                    skip_save = opt.skip_save,
                    ddim_steps = opt.ddim_steps,
                    fixed_code = opt.fixed_code,
                    ddim_eta = opt.ddim_eta,
                    n_iter = opt.n_iter,
                    H = opt.H,
                    W = opt.W,
                    C = opt.C,
                    f = opt.f,
                    n_samples = opt.n_samples,
                    n_rows = opt.n_rows,
                    scale = opt.scale,
                    from_file = opt.from_file,
                    seed = opt.seed,
                    small_batch = opt.small_batch,
                    precision = opt.precision,
                    url = opt.url,
                    config = cls.defaultModelArgs.config,
                    ckpt = cls.defaultModelArgs.ckpt,
                    device = cls.defaultModelArgs.device
        ))
