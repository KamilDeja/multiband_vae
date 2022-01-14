import torch
from vae_experiments import training_functions
import copy


def train_multiband(args, models_definition, local_vae, curr_global_decoder, task_id, train_dataset_loader,
                    train_dataset_loader_big, class_table,
                    n_classes, device):

    if args.gen_load_pretrained_models:
        local_vae=torch.load(args.gen_pretrained_models_dir + f'model{task_id}_local_vae').to(device)
    else:
        if task_id == 0:
            n_epochs = args.gen_ae_epochs + args.global_dec_epochs
        else:
            n_epochs = args.gen_ae_epochs
        tmp_table = training_functions.train_local_generator(local_vae, dataset=args.dataset,
                                                             task_loader=train_dataset_loader,
                                                             task_id=task_id, n_classes=n_classes,
                                                             n_epochs=n_epochs, local_start_lr=args.local_lr,
                                                             scheduler_rate=args.local_scheduler_rate,
                                                             scale_local_lr=args.scale_local_lr,
                                                             scale_marginal_loss=args.scale_reconstruction_loss,
                                                             use_lap_loss=args.lap_loss)
        class_table[task_id] = tmp_table
    print("Done training local VAE model")

    if not task_id:
        # First task, initializing global decoder as local_vae's decoder
        curr_global_decoder = copy.deepcopy(local_vae.decoder)
    else:
        print("Train global VAE model")
        # Retraining global decoder with previous global decoder and new data
        if args.gen_load_pretrained_models:
            curr_global_decoder = torch.load(args.gen_pretrained_models_dir + f'model{task_id}_curr_decoder').to(device)
        else:
            curr_global_decoder = training_functions.train_global_decoder(curr_global_decoder=curr_global_decoder,
                                                                          local_vae=local_vae,
                                                                          task_id=task_id, class_table=class_table,
                                                                          n_iterations=len(train_dataset_loader),
                                                                          n_epochs=args.global_dec_epochs,
                                                                          batch_size=args.gen_batch_size,
                                                                          train_same_z=True,
                                                                          models_definition=models_definition,
                                                                          dataset=args.dataset,
                                                                          cosine_sim=args.cosine_sim,
                                                                          global_lr=args.global_lr,
                                                                          scheduler_rate=args.global_scheduler_rate,
                                                                          limit_previous_examples=args.limit_previous,
                                                                          warmup_rounds=args.global_warmup,
                                                                          train_loader=train_dataset_loader,
                                                                          train_dataset_loader_big=train_dataset_loader_big,
                                                                          num_current_to_compare=args.generations_for_switch,
                                                                          experiment_name=args.experiment_name,
                                                                          visualise_latent=args.visualise_latent)
    torch.cuda.empty_cache()

    return curr_global_decoder
