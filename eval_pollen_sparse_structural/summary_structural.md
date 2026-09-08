# Pollen Structure-Aware Reconstruction Evaluation

Metrics are foreground-aware. `*_cal` values use foreground linear intensity matching before metric computation, so they emphasize recovered structure rather than global brightness scale.


## pollen-fov1-2xX-32.5mW

| Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal | SSIM>=0.85 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sparse3d_3d_xyz_K2 | 3d | xyz | 0.4078 | 0.3387 | 0.3190 | 0.1170 | 23.25 | False |
| sparse3d_2p5d_xy_K2_mask_wiener | 2p5d | xy | 0.4056 | 0.3343 | 0.3171 | 0.1170 | 23.25 | False |
| sparse3d_3d_z_K2_wiener_mdc25 | 3d | z | 0.4032 | 0.3324 | 0.3182 | 0.1023 | 23.23 | False |
| sparse3d_3d_xy_K2_wiener | 3d | xy | 0.3998 | 0.3337 | 0.3158 | 0.0762 | 23.21 | False |
| sparse3d_3d_z_K2 | 3d | z | 0.3983 | 0.3348 | 0.3158 | 0.0612 | 23.20 | False |
| sparse3d_3d_xyz_K2_wiener | 3d | xyz | 0.3979 | 0.3355 | 0.3166 | 0.0539 | 23.20 | False |
| sparse3d_2p5d_xyz_K2_mask_wiener_mdc25 | 2p5d | xyz | 0.3962 | 0.3346 | 0.3160 | 0.0444 | 23.19 | False |
| sparse3d_3d_z_K2_wiener | 3d | z | 0.3960 | 0.3357 | 0.3163 | 0.0389 | 23.19 | False |
| sparse3d_3d_z_K2_wiener_norm255_ft2 | 3d | z | 0.3959 | 0.3357 | 0.3164 | 0.0381 | 23.19 | False |
| sparse3d_3d_z_K2_wiener_e10_ref | 3d | z | 0.3948 | 0.3356 | 0.3162 | 0.0297 | 23.19 | False |
| sparse3d_2p5d_z_K2 | 2p5d | z | 0.3937 | 0.3357 | 0.3163 | 0.0209 | 23.19 | False |
| sparse3d_3d_z_K2_wiener_norm255 | 3d | z | 0.3930 | 0.3359 | 0.3164 | 0.0145 | 23.19 | False |
| sparse3d_2d_xy_K2_wiener | 2d | xy | 0.3930 | 0.3358 | 0.3162 | 0.0145 | 23.19 | False |
| sparse3d_2d_z_K2 | 2d | z | 0.3925 | 0.3359 | 0.3164 | 0.0107 | 23.19 | False |
| sparse3d_3d_xy_K2 | 3d | xy | 0.3920 | 0.3360 | 0.3164 | 0.0061 | 23.19 | False |
| sparse3d_2p5d_xy_K2_mask_wiener_mdc25 | 2p5d | xy | 0.3920 | 0.3360 | 0.3164 | 0.0060 | 23.19 | False |
| sparse3d_2p5d_xy_K2_mask | 2p5d | xy | 0.3919 | 0.3360 | 0.3164 | 0.0052 | 23.19 | False |
| sparse3d_2d_xy_K2 | 2d | xy | 0.3917 | 0.3360 | 0.3164 | 0.0037 | 23.19 | False |
| sparse3d_2p5d_xyz_K2 | 2p5d | xyz | 0.3916 | 0.3360 | 0.3164 | 0.0028 | 23.19 | False |
| sparse3d_2d_xyz_K2 | 2d | xyz | 0.3914 | 0.3360 | 0.3164 | 0.0014 | 23.19 | False |
| sparse3d_3d_xyz_K2_wiener_mdc25 | 3d | xyz | 0.3908 | 0.3360 | 0.3164 | -0.0039 | 23.19 | False |
| sparse3d_3d_xy_K2_wiener_mdc25 | 3d | xy | 0.3903 | 0.3359 | 0.3164 | -0.0073 | 23.19 | False |
| sparse3d_2d_xy_K2_wiener_mdc25 | 2d | xy | 0.3899 | 0.3359 | 0.3164 | -0.0107 | 23.19 | False |
| sparse3d_2d_xyz_K2_wiener_mdc25 | 2d | xyz | 0.3899 | 0.3359 | 0.3164 | -0.0107 | 23.19 | False |
| sparse3d_2d_z_K2_wiener_mdc25 | 2d | z | 0.3899 | 0.3359 | 0.3164 | -0.0107 | 23.19 | False |
| sparse3d_2p5d_xy_K2 | 2p5d | xy | 0.3898 | 0.3360 | 0.3165 | -0.0112 | 23.19 | False |
| sparse3d_2p5d_z_K2_mask_wiener_mdc25 | 2p5d | z | 0.3871 | 0.3354 | 0.3166 | -0.0318 | 23.19 | False |

## pollen-fov1-2xXY-32.5mW

| Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal | SSIM>=0.85 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sparse3d_3d_xy_K2 | 3d | xy | 0.6245 | 0.5343 | 0.6003 | 0.6009 | 25.13 | False |
| sparse3d_3d_xyz_K2 | 3d | xyz | 0.6170 | 0.5279 | 0.5950 | 0.5786 | 24.96 | False |
| sparse3d_3d_xyz_K2_wiener | 3d | xyz | 0.6144 | 0.5278 | 0.5940 | 0.5631 | 24.84 | False |
| sparse3d_2p5d_xy_K2_mask | 2p5d | xy | 0.5938 | 0.5007 | 0.5604 | 0.5542 | 24.78 | False |
| sparse3d_2p5d_xy_K2_mask_wiener | 2p5d | xy | 0.5664 | 0.4718 | 0.5270 | 0.5048 | 24.46 | False |
| sparse3d_3d_xy_K2_wiener | 3d | xy | 0.5639 | 0.4703 | 0.5201 | 0.5037 | 24.46 | False |
| sparse3d_2d_z_K2 | 2d | z | 0.5465 | 0.4465 | 0.5146 | 0.4588 | 24.21 | False |
| sparse3d_2d_xy_K2_wiener | 2d | xy | 0.4766 | 0.3865 | 0.4194 | 0.3016 | 23.61 | False |
| sparse3d_2d_xyz_K2 | 2d | xyz | 0.4724 | 0.3867 | 0.4079 | 0.2911 | 23.58 | False |
| sparse3d_2d_xy_K2 | 2d | xy | 0.4615 | 0.3699 | 0.3966 | 0.2813 | 23.55 | False |
| sparse3d_2p5d_xyz_K2 | 2p5d | xyz | 0.4266 | 0.3375 | 0.3480 | 0.2091 | 23.38 | False |
| sparse3d_3d_z_K2 | 3d | z | 0.4199 | 0.3357 | 0.3377 | 0.1829 | 23.33 | False |
| sparse3d_3d_z_K2_wiener_norm255_ft2 | 3d | z | 0.4099 | 0.3390 | 0.3273 | 0.1159 | 23.24 | False |
| sparse3d_3d_z_K2_wiener_e10_ref | 3d | z | 0.4083 | 0.3326 | 0.3240 | 0.1302 | 23.26 | False |
| sparse3d_3d_z_K2_wiener | 3d | z | 0.4083 | 0.3384 | 0.3258 | 0.1082 | 23.24 | False |
| sparse3d_2p5d_z_K2 | 2p5d | z | 0.4039 | 0.3339 | 0.3225 | 0.0943 | 23.22 | False |
| sparse3d_3d_z_K2_wiener_norm255 | 3d | z | 0.4009 | 0.3332 | 0.3179 | 0.0827 | 23.21 | False |
| sparse3d_2p5d_xyz_K2_mask_wiener_mdc25 | 2p5d | xyz | 0.3955 | 0.3352 | 0.3167 | 0.0359 | 23.19 | False |
| sparse3d_2p5d_xy_K2 | 2p5d | xy | 0.3939 | 0.3363 | 0.3173 | 0.0186 | 23.19 | False |
| sparse3d_2p5d_xy_K2_mask_wiener_mdc25 | 2p5d | xy | 0.3936 | 0.3358 | 0.3165 | 0.0191 | 23.19 | False |
| sparse3d_3d_xyz_K2_wiener_mdc25 | 3d | xyz | 0.3905 | 0.3360 | 0.3164 | -0.0063 | 23.19 | False |
| sparse3d_3d_xy_K2_wiener_mdc25 | 3d | xy | 0.3898 | 0.3359 | 0.3165 | -0.0112 | 23.19 | False |
| sparse3d_3d_z_K2_wiener_mdc25 | 3d | z | 0.3874 | 0.3357 | 0.3173 | -0.0314 | 23.19 | False |
| sparse3d_2p5d_z_K2_mask_wiener_mdc25 | 2p5d | z | 0.3858 | 0.3349 | 0.3164 | -0.0405 | 23.19 | False |
| sparse3d_2d_xy_K2_wiener_mdc25 | 2d | xy | 0.3849 | 0.3358 | 0.3191 | -0.0557 | 23.20 | False |
| sparse3d_2d_xyz_K2_wiener_mdc25 | 2d | xyz | 0.3849 | 0.3358 | 0.3191 | -0.0557 | 23.20 | False |
| sparse3d_2d_z_K2_wiener_mdc25 | 2d | z | 0.3849 | 0.3358 | 0.3191 | -0.0557 | 23.20 | False |

## pollen-fov1-2xY-32.5mW

| Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal | SSIM>=0.85 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sparse3d_3d_xyz_K2 | 3d | xyz | 0.7364 | 0.6754 | 0.7559 | 0.7003 | 26.17 | False |
| sparse3d_2p5d_xy_K2_mask_wiener | 2p5d | xy | 0.6351 | 0.5384 | 0.6320 | 0.6018 | 25.37 | False |
| sparse3d_3d_z_K2_wiener_e10_ref | 3d | z | 0.5279 | 0.4066 | 0.4620 | 0.5306 | 24.62 | False |
| sparse3d_3d_xyz_K2_wiener | 3d | xyz | 0.5192 | 0.4186 | 0.4646 | 0.4298 | 24.24 | False |
| sparse3d_3d_xy_K2_wiener | 3d | xy | 0.4929 | 0.3778 | 0.4480 | 0.3926 | 23.92 | False |
| sparse3d_2d_xyz_K2 | 2d | xyz | 0.4871 | 0.3727 | 0.4289 | 0.4000 | 23.94 | False |
| sparse3d_3d_z_K2_wiener_norm255_ft2 | 3d | z | 0.4829 | 0.3805 | 0.4095 | 0.3825 | 23.87 | False |
| sparse3d_3d_z_K2_wiener | 3d | z | 0.4820 | 0.3791 | 0.4071 | 0.3846 | 23.88 | False |
| sparse3d_2d_xy_K2_wiener | 2d | xy | 0.4754 | 0.3628 | 0.4173 | 0.3659 | 23.81 | False |
| sparse3d_3d_z_K2 | 3d | z | 0.4688 | 0.3655 | 0.3717 | 0.3920 | 23.91 | False |
| sparse3d_2d_xy_K2 | 2d | xy | 0.4398 | 0.3419 | 0.3624 | 0.2672 | 23.51 | False |
| sparse3d_2p5d_xyz_K2 | 2p5d | xyz | 0.4300 | 0.3383 | 0.3495 | 0.2291 | 23.42 | False |
| sparse3d_2p5d_z_K2 | 2p5d | z | 0.4180 | 0.3304 | 0.3381 | 0.1839 | 23.33 | False |
| sparse3d_2p5d_xy_K2 | 2p5d | xy | 0.4164 | 0.3372 | 0.3335 | 0.1599 | 23.30 | False |
| sparse3d_3d_z_K2_wiener_norm255 | 3d | z | 0.4142 | 0.3291 | 0.3320 | 0.1708 | 23.31 | False |
| sparse3d_2p5d_xyz_K2_mask_wiener_mdc25 | 2p5d | xyz | 0.4133 | 0.3213 | 0.3276 | 0.1955 | 23.35 | False |
| sparse3d_2d_z_K2 | 2d | z | 0.4124 | 0.3305 | 0.3288 | 0.1584 | 23.30 | False |
| sparse3d_2p5d_z_K2_mask_wiener_mdc25 | 2p5d | z | 0.4072 | 0.3229 | 0.3199 | 0.1593 | 23.30 | False |
| sparse3d_3d_xy_K2 | 3d | xy | 0.4026 | 0.3347 | 0.3197 | 0.0870 | 23.22 | False |
| sparse3d_2p5d_xy_K2_mask | 2p5d | xy | 0.4002 | 0.3351 | 0.3193 | 0.0684 | 23.21 | False |
| sparse3d_2p5d_xy_K2_mask_wiener_mdc25 | 2p5d | xy | 0.3994 | 0.3329 | 0.3179 | 0.0718 | 23.21 | False |
| sparse3d_3d_xy_K2_wiener_mdc25 | 3d | xy | 0.3940 | 0.3353 | 0.3164 | 0.0243 | 23.19 | False |
| sparse3d_3d_xyz_K2_wiener_mdc25 | 3d | xyz | 0.3928 | 0.3358 | 0.3164 | 0.0132 | 23.19 | False |
| sparse3d_2d_xy_K2_wiener_mdc25 | 2d | xy | 0.3853 | 0.3363 | 0.3179 | -0.0516 | 23.20 | False |
| sparse3d_2d_xyz_K2_wiener_mdc25 | 2d | xyz | 0.3853 | 0.3363 | 0.3179 | -0.0516 | 23.20 | False |
| sparse3d_2d_z_K2_wiener_mdc25 | 2d | z | 0.3853 | 0.3363 | 0.3179 | -0.0516 | 23.20 | False |
| sparse3d_3d_z_K2_wiener_mdc25 | 3d | z | 0.3849 | 0.3380 | 0.3234 | -0.0714 | 23.21 | False |

## pollen-fov1-4xX-32.5mW

| Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal | SSIM>=0.85 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sparse3d_3d_xy_K4_wiener | 3d | xy | 0.4394 | 0.3567 | 0.3583 | 0.2278 | 23.42 | False |
| sparse3d_2p5d_xy_K4 | 2p5d | xy | 0.4067 | 0.3329 | 0.3184 | 0.1273 | 23.26 | False |
| sparse3d_2p5d_z_K4 | 2p5d | z | 0.3944 | 0.3358 | 0.3168 | 0.0254 | 23.19 | False |
| sparse3d_3d_xy_K4 | 3d | xy | 0.3913 | 0.3360 | 0.3164 | 0.0005 | 23.19 | False |
| sparse3d_3d_xyz_K4 | 3d | xyz | 0.3913 | 0.3360 | 0.3164 | 0.0002 | 23.19 | False |
| sparse3d_2d_xy_K4 | 2d | xy | 0.3912 | 0.3360 | 0.3164 | -0.0005 | 23.19 | False |
| sparse3d_2d_xyz_K4 | 2d | xyz | 0.3912 | 0.3360 | 0.3164 | -0.0005 | 23.19 | False |
| sparse3d_2d_z_K4 | 2d | z | 0.3912 | 0.3360 | 0.3164 | -0.0005 | 23.19 | False |
| sparse3d_3d_z_K4 | 3d | z | 0.3911 | 0.3360 | 0.3164 | -0.0013 | 23.19 | False |
| sparse3d_2p5d_xyz_K4 | 2p5d | xyz | 0.3908 | 0.3360 | 0.3164 | -0.0037 | 23.19 | False |

## pollen-fov1-4xXY-32.5mW

| Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal | SSIM>=0.85 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sparse3d_3d_xy_K4_wiener | 3d | xy | 0.4638 | 0.3756 | 0.3902 | 0.2932 | 23.58 | False |
| sparse3d_2p5d_xyz_K4 | 2p5d | xyz | 0.4027 | 0.3360 | 0.3205 | 0.0827 | 23.21 | False |
| sparse3d_2p5d_z_K4 | 2p5d | z | 0.3946 | 0.3358 | 0.3169 | 0.0264 | 23.19 | False |
| sparse3d_3d_z_K4 | 3d | z | 0.3910 | 0.3360 | 0.3164 | -0.0018 | 23.19 | False |
| sparse3d_3d_xy_K4 | 3d | xy | 0.3909 | 0.3360 | 0.3164 | -0.0028 | 23.19 | False |
| sparse3d_3d_xyz_K4 | 3d | xyz | 0.3909 | 0.3360 | 0.3164 | -0.0030 | 23.19 | False |
| sparse3d_2d_xy_K4 | 2d | xy | 0.3909 | 0.3360 | 0.3164 | -0.0031 | 23.19 | False |
| sparse3d_2d_xyz_K4 | 2d | xyz | 0.3909 | 0.3360 | 0.3164 | -0.0031 | 23.19 | False |
| sparse3d_2d_z_K4 | 2d | z | 0.3908 | 0.3360 | 0.3164 | -0.0033 | 23.19 | False |
| sparse3d_2p5d_xy_K4 | 2p5d | xy | 0.3861 | 0.3357 | 0.3164 | -0.0402 | 23.19 | False |

## pollen-fov1-4xY-32.5mW

| Run | Algo | TrainSparse | Score | SSIM_fg_cal | MS-SSIM_fg_cal | NCC_fg | PSNR_fg_cal | SSIM>=0.85 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sparse3d_3d_xy_K4_wiener | 3d | xy | 0.4603 | 0.3729 | 0.3867 | 0.2825 | 23.55 | False |
| sparse3d_2p5d_z_K4 | 2p5d | z | 0.3957 | 0.3358 | 0.3174 | 0.0346 | 23.19 | False |
| sparse3d_2p5d_xyz_K4 | 2p5d | xyz | 0.3952 | 0.3365 | 0.3173 | 0.0285 | 23.19 | False |
| sparse3d_3d_xy_K4 | 3d | xy | 0.3912 | 0.3360 | 0.3164 | -0.0000 | 23.19 | False |
| sparse3d_3d_xyz_K4 | 3d | xyz | 0.3912 | 0.3360 | 0.3164 | -0.0003 | 23.19 | False |
| sparse3d_3d_z_K4 | 3d | z | 0.3910 | 0.3360 | 0.3164 | -0.0020 | 23.19 | False |
| sparse3d_2d_xy_K4 | 2d | xy | 0.3909 | 0.3360 | 0.3164 | -0.0029 | 23.19 | False |
| sparse3d_2d_xyz_K4 | 2d | xyz | 0.3909 | 0.3360 | 0.3164 | -0.0029 | 23.19 | False |
| sparse3d_2d_z_K4 | 2d | z | 0.3908 | 0.3360 | 0.3164 | -0.0032 | 23.19 | False |
| sparse3d_2p5d_xy_K4 | 2p5d | xy | 0.3833 | 0.3373 | 0.3179 | -0.0715 | 23.21 | False |
