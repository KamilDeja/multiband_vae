import torch
from vae_experiments import training_functions
import copy


def train_multiband(args, models_definition, local_vae, curr_global_decoder, task_id, train_dataset_loader, class_table,
                    n_classes, device):
    # if task_id > 0:
    #     local_vae.decoder = copy.deepcopy(curr_global_decoder)
    if args.gen_load_pretrained_models:
        local_vae.load_state_dict(torch.load(args.gen_pretrained_models_dir + f'model{task_id}_local_vae'))
    else:
        tmp_table = training_functions.train_local_generator(local_vae, task_loader=train_dataset_loader,
                                                             task_id=task_id, n_classes=n_classes,
                                                             n_epochs=args.gen_ae_epochs)
        class_table[task_id] = tmp_table
    print("Done training local VAE model")

    if not task_id:
        # First task, initializing global decoder as local_vae's decoder
        curr_global_decoder = copy.deepcopy(local_vae.decoder)
    else:
        print("Train global VAE model")
        # Retraining global decoder with previous global decoder and local_vae
        if args.gen_load_pretrained_models:
            curr_global_decoder = models_definition.Decoder(latent_size=local_vae.latent_size, d=args.gen_d,
                                                            p_coding=local_vae.p_coding,
                                                            n_dim_coding=local_vae.n_dim_coding,
                                                            cond_p_coding=local_vae.cond_p_coding,
                                                            cond_n_dim_coding=local_vae.cond_n_dim_coding,
                                                            cond_dim=n_classes, device=local_vae.device).to(device)
            curr_global_decoder.load_state_dict(
                torch.load(args.gen_pretrained_models_dir + f'model{task_id}_curr_decoder'))
        else:
            curr_global_decoder = training_functions.train_global_decoder(curr_global_decoder=curr_global_decoder,
                                                                          local_vae=local_vae,
                                                                          task_id=task_id, class_table=class_table,
                                                                          n_iterations=len(train_dataset_loader),
                                                                          n_epochs=args.global_dec_epochs,
                                                                          batch_size=args.gen_batch_size,
                                                                          train_same_z=True,
                                                                          models_definition=models_definition)
        torch.cuda.empty_cache()

    return curr_global_decoder
