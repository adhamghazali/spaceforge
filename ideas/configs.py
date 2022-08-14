unet1 = Unet(
# unet for imagen
    cond_on_text = True,
    dim_mults = [1, 2, 3, 4],
    cond_dim=512,
    dim = 256,
    layer_attns = [False, True, True, True],
    layer_cross_attns = [False, True, True, True],
)

unets = [unet1]


unet2 = Unet(
    cond_on_text = True,
    dim_mults = [1, 2, 4, 8],
    dim = 128,
    cond_dim=512,
    layer_attns = [False, False, False, True],
    layer_cross_attns = [False, False, False, True],
    memory_efficient = True,
)

unet3 = Unet(
    cond_on_text = True,
    dim_mults = [1, 2, 4, 8],
    dim = 128,
    cond_dim=512,
    layer_attns = [False, False, False, True],
    layer_cross_attns = [False, False, False, True],
    memory_efficient = True,
)


imagen = ElucidatedImagen(
    unets = (unet1, unet2, unet3),
    text_encoder_name=text_encoder,
    image_sizes = (64, 256, 1024),
    cond_drop_prob = 0.1,
    random_crop_sizes = (None, 128, 256),
    num_sample_steps = (64, 32, 32), # number of sample steps - 64 for base unet, 32 for upsampler (just an example, have no clue what the optimal values are)
    sigma_min = 0.002,           # min noise level
    sigma_max = (80, 160, 160),       # max noise level, @crowsonkb recommends double the max noise level for upsampler
    sigma_data = 0.5,            # standard deviation of data distribution
    rho = 7,                     # controls the sampling schedule
    P_mean = -1.2,               # mean of log-normal distribution from which noise is drawn for training
    P_std = 1.2,                 # standard deviation of log-normal distribution from which noise is drawn for training
    S_churn = 80,                # parameters for stochastic sampling - depends on dataset, Table 5 in apper
    S_tmin = 0.05,
    S_tmax = 50,
    S_noise = 1.003,
    auto_normalize_img = True,
)