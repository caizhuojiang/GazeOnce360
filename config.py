
cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gaze_weight': 10.0,
    'gpu_train': True,
    'batch_size': 9,
    'ngpu': 1,
    'epoch': 150,
    'decay1': 120,
    'decay2': 140,
    'image_size': 512,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}
