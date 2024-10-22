# CIFAR-10

    # ResNet 164
    run main.py --gpu 0 --save_path ./result_resnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16,32,64),N=(18,18,18),multiplier=4) --trainer Cifar10Trainer
    # Wide ResNet k=8, N=16
    run main.py --gpu 0 --save_path ./result_wide_resnet_8_16 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((16-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer
    # Wide ResNet k=8, N=28
    run main.py --gpu 0 --save_path ./result_wide_resnet_8_28 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((28-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer
    # DenseNet: growth_rate=12, depth=100
    run main.py --gpu 0 --save_path ./result_densenet_12_100 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model densenet.DenselyConnectedCNN(10,block_num=3,block_size=int((100-4)/6.0),growth_rate=12) --trainer Cifar10Trainer --weight_decay 1.0e-4
    # SqueezeNet
    run main.py --gpu 0 --save_path ./result_squeezenet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model squeezenet.SqueezeNet(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224
    # AlexNet
    run main.py --gpu 0 --save_path ./result_alexnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model alexnet.AlexNet(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01
    # VGG A
    run main.py --gpu 0 --save_path ./result_vgg_a --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model vgg_a.VGGA(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01
    # ResNext: C=2, d=64
    run main.py --gpu 0 --save_path ./result_resnext_2_64 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnext.ResNext(10,block_num=(3,)*3,C=2,d=64,multiplier=4)
    # ShakeShake: 64
    run main.py --gpu 0 --save_path ./result_shakeshake_64 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model shakeshake.ShakeShake(10,out_channels=(64,64*2,64*4),N=(4,)*3,branch_num=2)

# CIFAR-100

    # ResNet 164
    run main.py --gpu 0 --save_path ./result_cifar100_resnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(100,out_channels=(16,32,64),N=(18,18,18),multiplier=4) --trainer Cifar100Trainer
    # Wide ResNet k=8, N=16
    run main.py --gpu 0 --save_path ./result_cifar100_wide_resnet_8_16 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(100,out_channels=(16*8,32*8,64*8),N=(int((16-4)/6),)*3,multiplier=4) --trainer Cifar100Trainer
    # Wide ResNet k=8, N=28
    run main.py --gpu 0 --save_path ./result_cifar100_wide_resnet_8_28 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(100,out_channels=(16*8,32*8,64*8),N=(int((28-4)/6),)*3,multiplier=4) --trainer Cifar100Trainer
    # DenseNet: growth_rate=12, depth=40
    run main.py --gpu 0 --save_path ./result_cifar100_densenet_12_100 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model densenet.DenselyConnectedCNN(100,block_num=3,block_size=int((100-4)/6.0),growth_rate=12) --trainer Cifar100Trainer
    # SqueezeNet
    run main.py --gpu 0 --save_path ./result_cifar100_squeezenet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model squeezenet.SqueezeNet(100) --trainer Cifar100Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224
    # AlexNet
    run main.py --gpu 0 --save_path ./result_cifar100_alexnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model alexnet.AlexNet(100) --trainer Cifar100Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01
    # VGG A
    run main.py --gpu 0 --save_path ./result_cifar100_vgg_a --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model vgg_a.VGGA(100) --trainer Cifar100Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01
    # ResNext: C=2, d=64
    run main.py --gpu 0 --save_path ./result_cifar100_resnext_2_64 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnext.ResNext(100,block_num=(3,)*3,C=2,d=64,multiplier=4)  --trainer Cifar100Trainer
    # ShakeShake: 64
    run main.py --gpu 0 --save_path ./result_cifat100_shakeshake_64 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model shakeshake.ShakeShake(100,out_channels=(64,64*2,64*4),N=(4,)*3,branch_num=2)  --trainer Cifar100Trainer

# CIFAR-10: Detect Hard Examples
    # ResNet 164
    run main_hard.py --gpu 0 --save_path ./result_hard_resnet --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16,32,64),N=(18,18,18),multiplier=4) --trainer Cifar10Trainer --howmany 1
    # Wide ResNet k=8, N=16
     run main_hard.py --gpu 0 --save_path ./result_hard_wide_resnet_8_16 --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((16-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer --howmany 1
    # Wide ResNet k=8, N=28
    run main_hard.py --gpu 0 --save_path ./result_hard_wide_resnet_8_28 --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((28-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer --howmany 1
    # DenseNet: growth_rate=12, depth=40
    run main_hard.py --gpu 0 --save_path ./result_hard_densenet_12_40 --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model densenet.DenselyConnectedCNN(10,block_num=3,block_size=int((40-2)/3.0),growth_rate=12) --trainer Cifar10Trainer --howmany 1
    # SqueezeNet
    run main_hard.py --gpu 0 --save_path ./result_hard_squeezenet --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model squeezenet.SqueezeNet(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --howmany 1
    # AlexNet
    run main_hard.py --gpu 0 --save_path ./result_hard_alexnet --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model alexnet.AlexNet(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01 --howmany 1
    # VGG A
    run main_hard.py --gpu 0 --save_path ./result_hard_vgg_a --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model vgg_a.VGGA(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01 --howmany 1
    # ResNext: C=2, d=64
    run main_hard.py --gpu 0 --save_path ./result_hard_resnext_2_64 --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model resnext.ResNext(10,block_num=(3,)*3,C=2,d=64,multiplier=4) --howmany 1
    # ShakeShake: 64
    run main_hard.py --gpu 0 --save_path ./result_hard_shakeshake_64 --train_batch_size 64 --test_batch_size 100 --start_epoch 0 --epochs 200 --model shakeshake.ShakeShake(10,out_channels=(64,64*2,64*4),N=(4,)*3,branch_num=2) --howmany 1
