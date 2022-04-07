import sys
sys.path.append('/home/wxf/minigpt_prj/minGPT')

from transformers import DataCollatorForLanguageModeling, HfArgumentParser, GPT2TokenizerFast, TrainingArguments
from pile_streaming import get_pile_dataset
from trainer_dataset import TrainDataset
import time

from mingpt.logging import get_logger, use_src_log_handler
use_src_log_handler("in_root_logger")
logger = get_logger(__name__)


# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments))
    (training_args, ) = parser.parse_args_into_dataclasses()

    ################################################################################
    ################################### dataset ####################################
    ################################################################################
    GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokens_to_add = 128 - (len(tokenizer) % 128)
    tokenizer.add_special_tokens({'additional_special_tokens': [f'〈special{i}〉' for i in range(tokens_to_add)]})

    # signature_validator = RSASignatureValidator()
    dataset = get_pile_dataset()
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=2048)

    train_dataset = TrainDataset(
        args=training_args,
        train_dataset=dataset,
        data_collator=collator)
    train_dataloader = train_dataset.get_train_dataloader()
    epoch_iterator = train_dataloader

    ################################################################################
    ################################### train ####################################
    ################################################################################
    for step, inputs in enumerate(epoch_iterator):
        logger.info(f'step: {step} starts')
        start = time.time()
        input_ids = inputs["input_ids"]
        logger.info(f"{input_ids.shape}")
        # print(f"{input_ids.shape}")
