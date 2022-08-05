##############################################################
######################### GPT Config  ########################
##############################################################
### see details slides:
### https://docs.google.com/presentation/d/1MMvLTtNEPu-yG4x2apf9Bjo7cUVhn7tBmb4pk1ikM4k/edit#slide=id.g127fa9ad830_0_3
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        """
        1. vocab_size
        2. block_size

        block_size example
        https://docs.google.com/presentation/d/1SGeL6FpTmMrPx6io5qjOI4IAF5PpwJMy796UBMVa68Q/edit#slide=id.g1237b006913_0_3
        """
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

### The setting is from paper, see slides:
### https://docs.google.com/presentation/d/1Uq87bAUv5UoKwc49avuPiA4kkXFC4ttZmPFYcOJDnaQ/edit#slide=id.g1235cf120b8_0_71


### GPT-3 Small, 125M params.
class GPT3SimulateSmallConfig(GPTConfig):
    n_layer = 1
    n_embd = 768
    n_head = 12
    # so, d_head = 768/12 = 64


### GPT-3 Small, 125M params.
class GPT3SmallConfig(GPTConfig):
    n_layer = 12
    n_embd = 768
    n_head = 12
    # so, d_head = 768/12 = 64


### GPT-3 Medium, 350M params.
class GPT3MediumConfig(GPTConfig):
    n_layer = 24
    n_embd = 1024
    n_head = 16
    # so, d_head = 1024/16 = 64


### GPT-3 Large 760M params.
class GPT3LargeConfig(GPTConfig):
    n_layer = 24
    n_embd = 1536
    n_head = 16
    # so, d_head = 1536/16 = 96


### GPT-3 XL 1.3B params.
### This one is confusing, see slides: https://docs.google.com/presentation/d/1Uq87bAUv5UoKwc49avuPiA4kkXFC4ttZmPFYcOJDnaQ/edit#slide=id.g1235cf120b8_0_71
### n_embd=2048, n_head = 24, d_head = 2048/16 = 128 but not 2048/24 = 85.33
class GPT3XLConfig(GPTConfig):
    n_layer = 24
    n_embd = 2048
    n_head = 16
    # so, d_head = 2048/24 = 85.33, but wrong.
    # change to 2048/16 = 128


### GPT-3 2.7B config setting, set less layers to do profiling. Otherwise, OOM.
class GPT3_2dot7B_Simulte_Config(GPTConfig):
    """ GPT """
    n_layer = 4
    n_embd = 2560
    n_head = 32
    # so, d_head = 2560/32 = 80


### GPT-3 2.7B params.
class GPT3_2dot7B_Config(GPTConfig):
    n_layer = 32
    n_embd = 2560
    n_head = 32
    # so, d_head = 2560/32 = 80


### GPT-3 6.7B params, config setting, set less layers to do profiling. Otherwise, OOM.
class GPT3_6dot7B_Simulte_Config(GPTConfig):
    n_layer = 2
    n_embd = 4096
    n_head = 32
    # so, d_head = 4096/32 = 128


### GPT-3 13B params, config setting, set less layers to do profiling. Otherwise, OOM.
class GPT3_13B_Simulte_Config(GPTConfig):
    n_layer = 2
    n_embd = 5120
    n_head = 32


class GPT3_13B_config(GPTConfig):
    n_layer = 40
    n_embd = 5120 ### this one is confusing, see slides: https://docs.google.com/presentation/d/1Uq87bAUv5UoKwc49avuPiA4kkXFC4ttZmPFYcOJDnaQ/edit#slide=id.g1235cf120b8_0_71
    n_head = 40
    ### d_head = 5120/40 = 128.5


### GPT-3 specification: https://www.notion.so/xiaofengwu/GPT-3-submodule-input-output-shape-shape-a26ba70de8684282afb8f9de01642d67
class GPT3_175B_config(GPTConfig):
    n_layer = 96
    n_embd = 12288
    n_head = 96
    # so, d_head = 128 = 12288/96


### GPT-3 175B params, config setting, set less layers to do profiling. Otherwise, OOM.
class GPT3_175B_Simulte_Config(GPTConfig):
    n_layer = 1
    n_embd = 12288
    n_head = 96
    # so, d_head = 128 = 12288/96
