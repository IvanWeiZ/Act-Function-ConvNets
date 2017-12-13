# CSC2515
~~~
<h2> Running resnet model on cifar10 </h2>

> python cifar10_main.py [options]

*Options*
--data_dir <str> : path to data directory where dataset is stored 
--model_dir <str> : path where TF model will be stored
--resnet_size <int> : size of resnet
--train_epochs <int> : number of training epochs
--epochs_per_eval <int> : how many epochs between each evaluation
--batch_size <int> : batch size
--data_format : 'channels_first' / 'channels_last' - 'channels_first' for GPU performance boost

Command to run:
> python python cifar10_main.py --model_dir /tmp/cifar10_resnet14_tanh_50e --data_format channels_first
