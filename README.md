# New Features
- Contact plane registration implementation
- CVAE models, including implementation of Mixture of Gaussian prior
- CVAE loss functions

# Training
```
python3 train_relativerotation.py configs/models/8-19-21-dgcnn_mog_predict_forward_kld.py configs/losses/cvae_btb_loss_config.py out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples out/canonical_pointclouds/test/fps_randomizenoiseTrue_numfps2_samples 
```

# Generating data
### SO(3) augmentation

```
python3 data_generation/1_generate_pointclouds_v2.py parts/main/bandu_train/ test --no_table --no_simulate

python3 data_generation/2_generate_fps_pointclouds_2.py out/canonical_pointclouds/test/canonical_pointcloud_samples 2 1
 
```

### Viewing sample pkl

```
python3 5_visualize_sample_pkl.py ~/bandu_v1_full_clean/out/canonical_pointclouds/test/canonical_pointcloud_samples/Egg\ v2/0.pkl
```


# Credits

- https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- https://github.com/FlyingGiraffe/vnn
- https://github.com/AntixK/PyTorch-VAE/tree/master/models