from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)



print('-'*60)
for param_name in dict(model.named_parameters()).keys():
    print(param_name)
print('-'*60)


# 是错的
# 参考: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L144
no_decay = ["bias", "LayerNorm.weight"]
# ['bias', 'gamma', 'beta', 'LayerNorm'] # from https://github.com/airsplay/vokenization/blob/master/vlm/run_vlm_distributed.py


decay_param_names = [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
# print(decay_param_names)
for d_p in decay_param_names:
    print(d_p)



# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": 0.01,
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
# ]

# print('-'*60)
# for param_name in dict(model.named_parameters()).keys():
#     print(param_name)
# print('-'*60)

# print(optimizer_grouped_parameters)

