# Multiband VAE

This is the code repository to the Multiband VAE - an unsupervised method for continual learning of geneartive models.
For more information, please refer to the paper:

## Evaluation from article:

#### MNIST
```
python main.py --experiment_name MNIST_example --dataset MNIST --gpuid 0 --num_batches 10 --gen_d=32 --gen_latent_size 8 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --skip_normalization --seed 13 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --fc --local_scheduler_rate 0.98 --global_scheduler_rate 0.98
```

With the dirichlet split (accordingly for other datasets):

```
 python main.py --experiment_name MNIST_example_dirichlet --dataset MNIST --gpuid 0 --num_batches 10 --gen_d=32 --gen_latent_size 8 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --skip_normalization --seed 13 --score_on_val --cosine_sim 0.95 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --fc --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --dirichlet 1
```

#### FashionMNIST

```
python main.py --experiment_name FashionMNIST_example --dataset FashionMNIST --gpuid 0 --num_batches 10 --gen_batch_size 64 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.98 --global_lr 0.001 --local_lr 0.0005 --binary_latent_size 4 --global_warmup 5 --fc --local_scheduler_rate 0.98 --scale_reconstruction_loss 5 --dirichlet 1 --global_scheduler_rate 0.98
```

#### Omniglot

```
python main.py --experiment_name Omniglot_example --dataset Omniglot --gpuid 0 --num_batches 20 --gen_batch_size 64 --gen_d=32 --gen_latent_size 16 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 6 --gen_p_coding 13 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 12 --global_warmup 5 --fc --local_scheduler_rate 0.98 --global_scheduler_rate 0.98 --scale_reconstruction_loss 10
```

#### FashionMNIST->MNIST

```
python main.py --experiment_name DoubleMNIST_example --dataset DoubleMNIST --gpuid 0 --num_batches 10 --gen_batch_size 64 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --global_warmup 5 --fc --local_scheduler_rate 0.99 --scale_reconstruction_loss 3
```

#### MNIST->FashionMNIST
```
python main.py --experiment_name DoubleMNIST_example_2 --dataset DoubleMNIST --gpuid 0 --num_batches 10 --gen_batch_size 64 --gen_d=32 --gen_latent_size 12 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 1 --global_lr 0.001 --local_lr 0.001 --binary_latent_size 4 --global_warmup 5  --fc --local_scheduler_rate 0.99 --scale_reconstruction_loss 3 --reverse
```

#### CelebA
```
python main.py --experiment_name CelebA_example --dataset CelebA --gpuid 1 --num_batches 5 --gen_batch_size 64 --gen_d=50 --gen_latent_size 32 --gen_ae_epochs 70 --global_dec_epochs 140 --no_class_remap --gen_n_dim_coding 4 --gen_p_coding 9 --skip_normalization --seed 13 --gen_cond_n_dim_coding 0 --score_on_val --cosine_sim 0.95 --global_lr 0.003 --local_lr 0.001 --binary_latent_size 8 --global_warmup 5
```
## Credits
Repository based on [Continual-Learning-Benchmark](https://github.com/GT-RIPL/Continual-Learning-Benchmark)
