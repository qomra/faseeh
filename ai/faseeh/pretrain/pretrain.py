

import os
import math
import time
import torch
import logging
from functools import partial
from contextlib import nullcontext

from .task import Task
from .model import Transformer, ModelArgs




class Pretrainer:
    def __init__(self, output_dir, **kwargs):
        self.out_dir = output_dir
        
        # Training settings
        self.eval_interval = kwargs.get('eval_interval', 1000)
        self.log_interval = kwargs.get('log_interval', 1)
        self.eval_iters = kwargs.get('eval_iters', 1000)
        self.eval_only = kwargs.get('eval_only', False)
        self.always_save_checkpoint = kwargs.get('always_save_checkpoint', True)
        self.init_from = kwargs.get('init_from', 'scratch')
        
        # Data settings
        self.batch_size = kwargs.get('batch_size', 1)
        self.max_seq_len = kwargs.get('max_seq_len', 2048)
        self.vocab_source = kwargs.get('vocab_source', 'mysam/maajim')
        self.vocab_size = kwargs.get('vocab_size', 32010)
        
        # Model architecture
        self.dim = kwargs.get('dim', 2048)
        self.n_layers = kwargs.get('n_layers', 32)
        self.n_heads = kwargs.get('n_heads', 32)
        self.n_kv_heads = kwargs.get('n_kv_heads', 32)
        self.multiple_of = kwargs.get('multiple_of', 256)
        self.dropout = kwargs.get('dropout', 0.0)
        
        # Optimizer settings
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 4)
        self.learning_rate = kwargs.get('learning_rate', 5e-4)
        self.max_iters = kwargs.get('max_iters', 222784)
        self.weight_decay = kwargs.get('weight_decay', 1e-1)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.95)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        
        # Learning rate schedule
        self.decay_lr = kwargs.get('decay_lr', True)
        self.warmup_iters = kwargs.get('warmup_iters', 1000)
        
        # System settings
        self.device = kwargs.get('device', 'cuda')
        self.dtype = kwargs.get('dtype', 'bfloat16')
        self.compile = kwargs.get('compile', True)

        # fixing some hyperparams to sensible defaults
        self.lr_decay_iters = self.max_iters  # should be ~= max_iters per Chinchilla
        self.min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        self.seed_offset = 0  # offset for random seed

        # print the configuration in pretty way
        for k, v in self.config.items():
            logging.info(f"{k}: {v}")

    @property
    def config(self):
        return {
        # Training settings
        'eval_interval': self.eval_interval,
        'log_interval': self.log_interval,
        'eval_iters': self.eval_iters,
        'eval_only': self.eval_only,
        'always_save_checkpoint': self.always_save_checkpoint,
        'init_from': self.init_from,
        
        # Data settings
        'batch_size': self.batch_size,
        'max_seq_len': self.max_seq_len,
        'vocab_source': self.vocab_source,
        'vocab_size': self.vocab_size,
        
        # Model architecture
        'dim': self.dim,
        'n_layers': self.n_layers,
        'n_heads': self.n_heads,
        'n_kv_heads': self.n_kv_heads,
        'multiple_of': self.multiple_of,
        'dropout': self.dropout,
        
        # Optimizer settings
        'gradient_accumulation_steps': self.gradient_accumulation_steps,
        'learning_rate': self.learning_rate,
        'max_iters': self.max_iters,
        'weight_decay': self.weight_decay,
        'beta1': self.beta1,
        'beta2': self.beta2,
        'grad_clip': self.grad_clip,
        
        # Learning rate schedule
        'decay_lr': self.decay_lr,
        'warmup_iters': self.warmup_iters,
        'lr_decay_iters': self.lr_decay_iters,
        'min_lr': self.min_lr,
        
        # System settings
        'device': self.device,
        'dtype': self.dtype,
        'compile': self.compile,
        
        # Output directory
        'out_dir': self.out_dir
        }

    def train(self,data_source):
        tokens_per_iter = self.gradient_accumulation_steps  * self.batch_size * self.max_seq_len

        logging.info(f"tokens per iteration will be: {tokens_per_iter:,}")
        logging.info(f"breaks down as: {self.gradient_accumulation_steps} processes * {self.batch_size} batch size * {self.max_seq_len} max seq len")
        
        os.makedirs(self.out_dir, exist_ok=True)

        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        device_type = "cuda" if "cuda" in self.device else "cpu"  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )

        # task-specific setup
        iter_batches = partial(
            Task.iter_batches,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            data_source=data_source,
            device=self.device,
            num_workers=0,
        )   
        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        # model init
        model_args = dict(
            dim=self.dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            vocab_size=self.vocab_size,
            multiple_of=self.multiple_of,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
        )  # start with model_args from command line

        if self.init_from == "scratch":
            # init a new model from scratch
            logging.info("Initializing a new model from scratch")
            gptconf = ModelArgs(**model_args)
            model = Transformer(gptconf)
        elif self.init_from == "resume":
            logging.info(f"Resuming training from {self.out_dir}")
            # resume training from a checkpoint.
            ckpt_path = os.path.join(self.out_dir, "ckpt.pt")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            checkpoint_model_args = checkpoint["model_args"]
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            gptconf = ModelArgs(**model_args)
            model = Transformer(gptconf)
            state_dict = checkpoint["model"]
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = "_orig_mod."
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint["iter_num"]
            best_val_loss = checkpoint["best_val_loss"]
        
        model.to(self.device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler = torch.amp.GradScaler(enabled=(self.dtype == "float16"))

        # optimizer
        optimizer = model.configure_optimizers(self.weight_decay, self.learning_rate, (self.beta1, self.beta2), device_type)
        if self.init_from == "resume" and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        checkpoint = None  # free up memory

        # compile the model
        if compile:
            logging.info("compiling the model... (takes a ~minute)")
            model = torch.compile(model)  # requires PyTorch 2.0

        # helps estimate an arbitrarily accurate loss over either split using many batches
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ["train", "val"]:
                batch_iter = iter_batches(split=split)
                losses = torch.zeros(self.eval_iters)  # keep on CPU
                for k in range(self.eval_iters):
                    X, Y = next(batch_iter)
                    with ctx:
                        logits = model(X, Y)
                        loss = raw_model.last_loss
                    losses[k] = loss.item()
                out[split] = losses.mean()
            model.train()
            return out
        # learning rate decay scheduler (cosine with warmup)
        def get_lr(it):
            # 1) linear warmup for warmup_iters steps
            if it < self.warmup_iters:
                return self.learning_rate * it / self.warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it > self.lr_decay_iters:
                return self.min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return self.min_lr + coeff * (self.learning_rate - self.min_lr)
        # training loop
        train_batch_iter = iter_batches(split="train")
        X, Y = next(train_batch_iter)  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = model  # unwrap DDP container if needed
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if self.decay_lr else self.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            losses = estimate_loss()

            if iter_num % self.eval_interval == 0:
                if losses["val"] < best_val_loss or self.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": model_args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "config": self.config,
                        }
                        logging.info(f"saving checkpoint to {self.out_dir}")
                        torch.save(checkpoint, os.path.join(self.out_dir, "ckpt.pt"))
                    
            if iter_num == 0 and self.eval_only:
                break
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.gradient_accumulation_steps):
                with ctx:
                    logits = model(X, Y)
                    loss = raw_model.last_loss
                    loss = loss / self.gradient_accumulation_steps
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = next(train_batch_iter)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if self.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_interval == 0 :
                # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(self.batch_size * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                logging.info(
                    f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.max_iters:
                break


        









    



















