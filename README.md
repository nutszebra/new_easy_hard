# CIFAR-10

    # ResNet 164
    run main.py --gpu 0 --save_path ./result_resnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16,32,64),N=(18,18,18),multiplier=4) --trainer Cifar10Trainer
    # Wide ResNet k=8, N=16
    run main.py --gpu 0 --save_path ./result_wide_resnet_8_16 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((16-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer
    # Wide ResNet k=8, N=28
    run main.py --gpu 0 --save_path ./result_wide_resnet_8_28 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((28-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer
    # DenseNet: growth_rate=12, depth=40
    run main.py --gpu 0 --save_path ./result_densenet_12_40 --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model densenet.DenselyConnectedCNN(10,block_num=3,block_size=int((40-2)/3.0),growth_rate=12) --trainer Cifar10Trainer
    # SqueezeNet
    run main.py --gpu 0 --save_path ./result_squeezenet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model squeezenet.SqueezeNet(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224
    # AlexNet
    run main.py --gpu 0 --save_path ./result_alexnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model alexnet.AlexNet(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01
    # VGG A
    run main.py --gpu 0 --save_path ./result_vgg_a --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model vgg_a.VGGA(10) --trainer Cifar10Trainer --train_transform transformers.train_cifar10_224 --test_transform transformers.test_cifar10_224 --lr 0.01


