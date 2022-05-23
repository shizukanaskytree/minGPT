import sys
sys.path.append('/home/wxf/minigpt_prj/minGPT')

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
debugpy.breakpoint()

import torch
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, GPT2TokenizerFast, TrainingArguments
from pile_streaming import get_pile_dataset
from trainer_dataset import TrainDataset
import time
import math

from mingpt.model import GPT
from mingpt.config import *
from mingpt.trainer import Trainer, TrainerConfig

from mingpt.logging import get_logger, use_src_log_handler
use_src_log_handler("in_root_logger")
logger = get_logger(__name__)

from scheduler import get_linear_schedule_with_warmup

if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments))
    (training_args, ) = parser.parse_args_into_dataclasses()

    ##############################################################################
    ################################### model ####################################
    ##############################################################################
    SEQUENCE_LENGTH = 2048
    mconf = GPTTestConfig(vocab_size=50304, block_size=SEQUENCE_LENGTH)
    model = GPT(mconf)
    n_paraams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"GPT has {n_paraams} params")

    ##############################################################################
    ################################ Trainer Config ##############################
    ##############################################################################
    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(
        max_epochs=2,
        batch_size=1,
        learning_rate=6e-4,
        lr_decay=True,
        warmup_tokens=512*20,
        # final_tokens=2*len(train_dataset)*block_size, # pile dataset has no len()
        num_workers=4)
    config = tconf

    ##############################################################################
    ################################### optimizer ################################
    ##############################################################################
    ### minigpt optimizer
    optimizer = model.configure_optimizers(config)
    # optimizer = model.config_lamb_optim()
    # optimizer = model.config_adam_optim()

    global_batch_size = 4096 # 1 sec 1 sample, too long

    ### Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
    ### https://arxiv.org/pdf/1909.08053.pdf
    num_training_steps = 15000 # the global step. 300k steps total
    num_warmup_steps = 0 # 3125    # the global warm up step, it takes a long time.

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lambda step: 1 - step/num_training_steps)

    ################################################################################
    ################################### dataset ####################################
    ################################################################################
    # GPT2TokenizerFast has the attribute vocab_size=50257
    GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokens_to_add = 128 - (len(tokenizer) % 128)
    tokenizer.add_special_tokens({'additional_special_tokens': [f'〈special{i}〉' for i in range(tokens_to_add)]})

    block_size = 2048 # why 512 is wrong?
    ### tokenizer.vocab_size+1 adjust for pile dataset
    # mconf = GPT3SimulteConfig(tokenizer.vocab_size+1, block_size=block_size)
    # Make the setting the same as the swarm experiment.
    # signature_validator = RSASignatureValidator()
    dataset = get_pile_dataset()

    # since we will slice the input with one token offset left and right
    pad_to_multiple_of = block_size + 1
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of)

    train_dataset = TrainDataset(
        args=training_args,
        train_dataset=dataset,
        data_collator=collator)
    train_dataloader = train_dataset.get_train_dataloader()

    ##############################################################################
    ################################### train ####################################
    ##############################################################################
    # take over whatever gpus are on the system
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        # self.model = torch.nn.DataParallel(self.model).to(self.device)
        model = model.to(device)

    tokens = 0 # counter used for learning rate decay

    best_loss = float('inf')

    is_train = True
    model.train(is_train)
    losses = []

    for step, inputs in enumerate(train_dataloader):
        # logger.info(f'step: {step} starts')
        start = time.time()
        input_ids = inputs["input_ids"]
        # logger.info(f"{input_ids.shape}")

        input_ids = input_ids.to(device)

        ### prepare input and label for the model
        shift_input_ids = input_ids[..., :-1].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # shift_input_ids.shape
        # torch.Size([1, 2048])
        # shift_labels.shape
        # torch.Size([1, 2048])

        ### forward the model
        with torch.set_grad_enabled(is_train):
            logits, loss = model(idx=shift_input_ids, targets=shift_labels)
            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())

            if is_train:
                loss.backward()
                if step % global_batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    # backprop and update the parameters
                    # Zero gradients, perform a backward pass, and update the weights.
                    # Before the backward pass, use the optimizer object to zero all of the
                    # gradients for the variables it will update (which are the learnable
                    # weights of the model). This is because by default, gradients are
                    # accumulated in buffers( i.e, not overwritten) whenever .backward()
                    # is called. Checkout docs of torch.autograd.backward for more details.
                    optimizer.zero_grad()
                    # scheduler.step() # change lr after a global step
                    # report progress
                    logger.info(f"{step} done, train loss {loss.item():.5f}")
                    # logger.info(f"{step} done, train loss {loss.item():.5f}, {scheduler.get_lr()}")

                # decay the learning rate based on our progress
                if config.lr_decay:
                    tokens += (shift_labels >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                    if tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate
