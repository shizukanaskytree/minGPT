from collections import defaultdict

import torch
from datasets import load_dataset, interleave_datasets, disable_progress_bar, set_progress_bar_enabled
from transformers import DataCollatorForLanguageModeling, GPT2TokenizerFast, AlbertTokenizerFast
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from mingpt.trainer import Trainer
from mingpt.model import GPT

GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
tokens_to_add = 128 - (len(tokenizer) % 128)
tokenizer.add_special_tokens({'additional_special_tokens': [f'〈special{i}〉' for i in range(tokens_to_add)]})

# tokenizer = AlbertTokenizerFast.from_pretrained("albert-large-v2")

MAX_SEQ_LENGTH = 2048

def split_list(l, n):
    # splits list/string into n size chunks
    return (l[i:i + n] for i in range(0, len(l), n))


def process_instance(text, max_seq_length):
    tokenized_text = tokenizer.encode(text) + [tokenizer.eos_token_id]

    for chunk in split_list(tokenized_text, max_seq_length):
        yield chunk


def examples_from_documents(documents, max_seq_length=MAX_SEQ_LENGTH):
    texts = (text for text in documents["text"] if len(text) > 0 and not text.isspace())

    new_examples = defaultdict(list)

    for text in texts:
        instances = process_instance(text, max_seq_length)

        for instance in instances:
            new_examples['input_ids'].append(instance)

    return new_examples


def get_wiki_train_dataset(seed):
    ### todo: change the var name
    pile = load_dataset(
                "wikitext",
                "wikitext-103-v1",
                # cache_dir="./data/cache",
                streaming=True,
                split="train")

    shuffled_wiki = pile.shuffle(buffer_size=100, seed=seed)
    tokenized_wiki = shuffled_wiki.map(examples_from_documents, batched=True, batch_size=4)
    tokenized_wiki = tokenized_wiki.with_format('torch')
    return tokenized_wiki


if __name__ == '__main__':
    # import debugpy; debugpy.listen(5678); debugpy.wait_for_client()

    #---------------------------------------------------------------------------
    # training dataset
    #---------------------------------------------------------------------------
    train_dataset = get_wiki_train_dataset(seed=1)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=MAX_SEQ_LENGTH)

    train_dataloader = DataLoader(
                train_dataset,
                batch_size=1,
                collate_fn=collator,
                num_workers=0, #self.args.dataloader_num_workers, # 1
                pin_memory=True, # from args default.
            )

    #---------------------------------------------------------------------------
    # model
    #---------------------------------------------------------------------------
    config_model = GPT.get_default_config()
    config_model.model_type = 'gpt2'
    print(f"len(tokenizer) = {len(tokenizer)}")

    config_model.vocab_size = len(tokenizer)
    config_model.block_size = MAX_SEQ_LENGTH
    model = GPT(config_model)

    #---------------------------------------------------------------------------
    # trainer boilerplate and setup adam optimizer
    #---------------------------------------------------------------------------
    config_trainer = Trainer.get_default_config()
    trainer = Trainer(config_trainer, model, train_dataloader)

    #---------------------------------------------------------------------------
    # iteration callback
    #---------------------------------------------------------------------------
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        # if trainer.iter_num % 500 == 0:
        #     # evaluate both the train and test score
        #     model.eval()
        #     with torch.no_grad():
        #         # sample from the model...
        #         context = "O God, O God!"
        #         x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
        #         y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
        #         completion = ''.join([train_dataset.itos[int(i)] for i in y])
        #         print(completion)
        #     # save the latest model
        #     print("saving model")
        #     ckpt_path = os.path.join(config.system.work_dir, "model.pt")
        #     torch.save(model.state_dict(), ckpt_path)
        #     # revert model to training mode
        #     model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    #---------------------------------------------------------------------------
    # run training
    #---------------------------------------------------------------------------
    trainer.run()

