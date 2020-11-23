## Example usage on MNIST
```
python main.py --experiment_name MNIST_experiment --seed 12 --gpuid 0 --dataset MNIST --first_split_size 2 --other_split_size 2 --base_force_out_dim 10 --base_model_type lenet --base_model_name LeNetG --gen_batch_size 40 --base_batch_size 120 --base_schedule 15 --base_lr 0.002 --no_class_remap --skip_normalization --new_task_data_processing generated --gen_ae_pre_epochs 20 --gen_ae_epochs 200
```

## Credits
Forked from [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark)
