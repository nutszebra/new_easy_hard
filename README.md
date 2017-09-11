# CIFAR-10

    # ResNet 164
    run main.py --gpu 0 --save_path ./result_resnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16,32,64),N=(18,18,18),multiplier=4) --trainer Cifar10Trainer
    # Wide ResNet k=8, N=16
    run main.py --gpu 0 --save_path ./result_resnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((16-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer
    # Wide ResNet k=8, N=28
    run main.py --gpu 0 --save_path ./result_resnet --train_batch_size 64 --test_batch_size 100 --start_epoch 1 --epochs 200 --model resnet.ResidualNetwork(10,out_channels=(16*8,32*8,64*8),N=(int((28-4)/6),)*3,multiplier=4) --trainer Cifar10Trainer

