class MODEL :
    in_dim = 3+3
    hid_dim = 512
    out_dim = 3+2 # [xyz, vel, cont]

    up_sample_rate = 3
    down_sample_rate = 2
    res_depth = 2
    dilation_growth_rate = 3

    div_rate = 4
class CODEBOOK:
    nb_code = 512
    code_dim = 64