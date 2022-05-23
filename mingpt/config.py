class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPTSmallConfig(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_embd = 768
    n_head = 12

# The setting is from paper, see slides:
# https://docs.google.com/presentation/d/1Uq87bAUv5UoKwc49avuPiA4kkXFC4ttZmPFYcOJDnaQ/edit#slide=id.g1235cf120b8_0_71
class GPT3SmallConfig(GPTConfig):
    """ GPT params """
    n_layer = 12
    n_embd = 768
    n_head = 12
    # so, d_head = 768/12 = 64

class GPT3MediumConfig(GPTConfig):
    """ GPT: 302,575,616 params """
    n_layer = 24
    n_embd = 1024
    n_head = 16
    # so, d_head = 768/12 = 64

class GPT3LargeConfig(GPTConfig):
    """ GPT: 680,355,840 params, 3792MiB """
    n_layer = 24
    n_embd = 1536
    n_head = 16
    # so, d_head = 768/12 = 96


class GPT3XLConfig(GPTConfig):
    """ GPT: 1,209,131,008 params, 5712MiB """
    n_layer = 24
    n_embd = 2048
    n_head = 16
    # so, d_head = 768/12 = 128

class GPT3_2dot7B_Config(GPTConfig):
    """ GPT: 2,518,312,960 params, 10854MiB """
    n_layer = 32
    n_embd = 2560
    n_head = 32
    # so, d_head = 768/12 = 80


class GPT3_2dot7B_Simulte_Config(GPTConfig):
    """ GPT: 577,274,880 params, 19206MiB """
    n_layer = 4 # 1 transformer block = 78,676,480 params, since 5 blocks is (655,951,360 -  577,274,880), 22366MiB
    n_embd = 2560
    n_head = 32
    # so, d_head = 768/12 = 80

class GPT3_6dot7B_Simulte_Config(GPTConfig):
    """ GPT: 20482MiB """
    n_layer = 2 # 32 initially
    n_embd = 4096
    n_head = 32
    # so, d_head = 128

    """
    number of parameters: 8.228700e+08
    per gpu: 205,717,504

    n_params_tok_emb: 205,856,768
    n_params_pos_emb:   8,388,608

    n_params_blocks:    402,759,680
    n_params_per_block: 201,379,840

    n_params_ln_f: 8192
    n_params_head: 205,856,768

    ##########################################
    ################ 可以做实验 ################
    ##########################################
    和原来 albert 的分配 layers 一样.
    head: 1/4 transformer blocks + n_params_tok_emb + n_params_pos_emb
    body1: 1/4 transformer blocks
    body2: 1/4 transformer blocks
    tail: 1/4 transformer blocks + n_params_ln_f + n_params_head

    32 / 4 = 8
    head: 8 blocks + n_params_tok_emb + n_params_pos_emb ~= 9 blocks
    body1: 8 blocks
    body2: 8 blocks
    tail: 8 blocks + n_params_ln_f + n_params_head ~= 9 blocks

    OOM
    ----------------------------------------
    16 blocks
    最多 4 blocks each partition

    head: 4 blocks + n_params_tok_emb + n_params_pos_emb ~= 5 blocks
    body1: 4 blocks
    body2: 4 blocks
    tail: 4 blocks + n_params_ln_f + n_params_head ~= 5 blocks

    """

class GPT3SimulteConfig(GPTConfig):
    """ GPT:
    n_layer=2, #params 4,884,529,152, OOM
    """
    n_layer = 1
    n_embd = 12288
    n_head = 96
    # so, d_head = 768/12 = 80

    """
    不可以做实验, limit:
    32 GB 可以存放的 nparams ~= 1 B
    30836MiB ~ number of parameters: 1.083230e+09

    4 V100 model parallel limit: 4B model

    number of parameters: 3.072430e+09
    per gpu: 768,107,520
    n_params_tok_emb: 617,570,304
    n_params_pos_emb: 25,165,824
    n_params_blocks: 1,812,099,072
    n_params_per_block: 1,812,099,072
    n_params_ln_f: 24,576
    n_params_head: 617,570,304

    Block:
    n_params_attn: 604,028,928
    n_params_mlp: 1,208,020,992
    n_params_ln1: 24576
    n_params_ln2: 24576
    """

class GPTTestConfig(GPTConfig): # GPT3_6dot7B_Simulte_Config
    """ large """
    n_layer = 24
    n_embd = 1536
    n_head = 16
