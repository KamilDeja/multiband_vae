from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from types import MethodType
import models
import copy
from utils.metric import accuracy, AverageMeter, Timer
from vae_experiments import vae_utils


class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''

    def __init__(self, agent_config, **kw):
        '''
        :param agent_config (dict): lr=float,momentum=float,weight_decay=float,
                                    base_schedule=[int],  # The last number in the list is the end of epoch
                                    base_model_type=str,base_model_name=str,out_dim={task:dim},base_model_weights=str
                                    force_single_head=bool
                                    base_print_freq=int
                                    gpuid=[int]
        '''
        super(NormalNN, self).__init__()
        self.log = print if agent_config['base_print_freq'] > 0 else lambda \
                *args: None  # Use a void function to replace the print
        self.config = agent_config
        # If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
        self.multihead = True if len(
            self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
        self.model = self.create_model(**kw)

        self.score_generated_images_by_freezed_classifier = agent_config['score_generated_images_by_freezed_classifier']
        # self.criterion_fn = nn.CrossEntropyLoss()
        self.criterion_fn = nn.MSELoss()
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        self.init_optimizer()
        self.reset_optimizer = False
        self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
        # Set a interger here for the incremental class scenario

    def one_hot_targets(self, y, n_classes=10):
        if not y.nelement():
            return y

        zero_ar = torch.zeros(y.shape[0], n_classes)
        zero_ar[np.array(range(y.shape[0])), y] = 1.0
        return zero_ar

    def init_optimizer(self):
        optimizer_arg = {'params': self.model.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
            optimizer_arg['nesterov'] = self.config['nesterov']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'

        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)


    def create_model(self, **kw):
        cfg = self.config

        # Define the backbone (MLP, LeNet, VGG, ResNet ... etc) of model
        model = models.__dict__[cfg['base_model_type']].__dict__[cfg['base_model_name']](**kw)

        # Apply network surgery to the backbone
        # Create the heads for tasks (It can be single task or multi-task)
        n_feat = model.last.in_features

        # The output of the model will be a dict: {task_name1:output1, task_name2:output2 ...}
        # For a single-headed model the output will be {'All':output}
        model.last = nn.ModuleDict()
        for task, out_dim in cfg['out_dim'].items():
            model.last[task] = nn.Linear(n_feat, out_dim)

        # Redefine the task-dependent function
        def new_logits(self, x):
            outputs = {}
            for task, func in self.last.items():
                outputs[task] = func(x)
            return outputs

        # Replace the task-dependent function
        model.logits = MethodType(new_logits, model)
        # Load pre-trained weights
        if cfg['base_model_weights'] is not None:
            print('=> Load model weights:', cfg['base_model_weights'])
            model_state = torch.load(cfg['base_model_weights'],
                                     map_location=lambda storage, loc: storage)  # Load to CPU.
            model.load_state_dict(model_state, strict=False)
            print('=> Load Done')
        return model

    def forward(self, x):
        return self.model.forward(x)

    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        for t in out.keys():
            out[t] = out[t].detach()
        return out

    def validation(self, dataloader):
        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        batch_timer.tic()

        orig_mode = self.training
        self.eval()
        for i, (input, target, task) in enumerate(dataloader):
            target = self.one_hot_targets(target, self.model.n_classes)
            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            output = self.predict(input)

            # Summarize the performance of all tasks, or 1 task, depends on dataloader.
            # Calculated by total number of data.
            acc = accumulate_acc(output, target, task, acc)

        self.train(orig_mode)

        self.log(' * Val Acc {acc.avg:.6f}, Total time {time:.3f}'
                 .format(acc=acc, time=batch_timer.toc()))
        return acc.avg

    def criterion(self, preds, targets, tasks, **kwargs):
        # The inputs and targets could come from single task or a mix of tasks
        # The network always makes the predictions with all its heads
        # The criterion will match the head and task to calculate the loss.
        if self.multihead:
            loss = 0
            for t, t_preds in preds.items():
                inds = [i for i in range(len(tasks)) if tasks[i] == t]  # The index of inputs that matched specific task
                if len(inds) > 0:
                    t_preds = t_preds[inds]
                    t_target = targets[inds]
                    loss += self.criterion_fn(t_preds, t_target) * len(inds)  # restore the loss from average
            loss /= len(targets)  # Average the total loss by the mini-batch size
        else:
            pred = preds['All']
            if self.criterion_fn._get_name() == 'MSELoss':
                pred = F.softmax(pred, 1)
                if isinstance(self.valid_out_dim,
                              int):  # (Not 'ALL') Mask out the outputs of unseen classes for incremental class scenario
                    pred = preds['All'][:, :self.valid_out_dim]
                loss = self.criterion_fn(pred, targets)
            else:
                targets = torch.argmax(targets, 1)
                loss = self.criterion_fn(pred, targets)
        return loss

    def update_model(self, inputs, targets, tasks):
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, out

    def learn_batch(self, train_loader, val_loader=None, curr_global_decoder=None, local_vae=None, class_table=None,
                    global_classes_list=None, task_id=None, n_codes=None, global_n_codes=None,
                    new_task_data_processing='original'):
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()

        print("Classifier: learning new task in '{}' new data processing mode".format(
            new_task_data_processing))

        if new_task_data_processing == 'original':
            process_through_local_vae = False
            train_only_on_generated_data = False
        elif new_task_data_processing == 'original_through_vae':
            process_through_local_vae = True
            train_only_on_generated_data = False
        elif new_task_data_processing == 'generated':
            process_through_local_vae = False
            train_only_on_generated_data = True
        else:
            raise ValueError("'new_task_data_processing' argument is invalid: '{}'. "
                             "Valid values are: 'original', 'original_through_vae', 'generated.")

        if self.score_generated_images_by_freezed_classifier:
            frozen_model = copy.deepcopy(self.model)
            frozen_model.eval()

        train_accs = []
        val_accs = []

        for epoch in range(self.config['base_schedule'][-1]):
            data_timer = Timer()
            batch_timer = Timer()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            acc = AverageMeter()

            # Config the model and optimizer
            self.log('Epoch:{0}'.format(epoch))
            self.model.train()
            for param_group in self.optimizer.param_groups:
                self.log('LR:', param_group['lr'])

            # Learning with mini-batch
            data_timer.tic()
            batch_timer.tic()
            self.log('Itr\t\t  Time\t\t\t  Data\t\t\t  Loss\t\t\t  Acc')

            current_start = 0

            if train_only_on_generated_data:
                n_tasks_to_generate = task_id + 1
            else:
                n_tasks_to_generate = task_id

            if not train_only_on_generated_data and (task_id == 0):
                starting_points_fixed = np.array([[0]])
            else:
                starting_points = []
                for prev_task_id in range(n_tasks_to_generate):
                    starting_points.append(
                        np.random.permutation(
                            np.array(range(math.ceil(global_n_codes[prev_task_id] / train_loader.batch_size)))))
                max_len = max([len(repeats) for repeats in starting_points])
                starting_points_fixed = []
                for points in starting_points:
                    starting_points_fixed.append(np.pad(points, [0, max_len - len(points)], mode="reflect"))
                starting_points_fixed = np.array(starting_points_fixed)

            for i, (orig_input, orig_target, orig_task) in enumerate(train_loader):

                data_time.update(data_timer.toc())  # measure data loading time

                batch_size = len(orig_task)

                # generate data so every task is equally represented
                with torch.no_grad():
                    if process_through_local_vae:
                        orig_input, orig_target, _ = vae_utils.generate_current_data(local_vae.decoder, task_id,
                                                                                     batch_size, current_start,
                                                                                     global_classes_list, n_codes,
                                                                                     global_n_codes)

                    generate_impl = vae_utils.generate_previous_data

                    if train_only_on_generated_data:
                        # generate data from previous tasks and the current one
                        generate_impl = vae_utils.generate_previous_and_current_data
                        # clear original data
                        orig_input, orig_target = torch.Tensor(), torch.Tensor()

                    if train_only_on_generated_data or (task_id>0):
                        gen_input, gen_target_orig, _ = generate_impl(curr_global_decoder, task_id, batch_size,
                                                                      starting_points_fixed[:, current_start] * batch_size,
                                                                      global_classes_list, n_codes, global_n_codes)
                        current_start += 1
                    else:
                        gen_input = torch.Tensor()
                        gen_target_orig = torch.Tensor()

                    if self.score_generated_images_by_freezed_classifier:
                        if task_id > 0:
                            gen_target = frozen_model.forward(gen_input[:-batch_size])
                            gen_target = gen_target['All']
                            gen_target = F.softmax(gen_target, 1)

                            if train_only_on_generated_data:
                                targets_orig = self.one_hot_targets(gen_target_orig[-batch_size:]).to(local_vae.device)
                                gen_target = torch.cat([gen_target, targets_orig])
                            else:
                                targets_orig = self.one_hot_targets(orig_target).to(local_vae.device)
                                gen_target = torch.cat([gen_target, targets_orig])
                        else:
                            gen_target = gen_target_orig
                            gen_target = self.one_hot_targets(gen_target, self.model.n_classes)
                    else:
                        gen_target = self.one_hot_targets(gen_target, self.model.n_classes)

                orig_target = self.one_hot_targets(orig_target, self.model.n_classes)
                if self.gpu:
                    orig_input = orig_input.cuda()
                    orig_target = orig_target.cuda()
                    gen_input = gen_input.cuda()
                    gen_target = gen_target.cuda()

                # merge original and generated data
                multi_input = torch.cat((orig_input, gen_input), 0)
                multi_target = torch.cat((orig_target, gen_target), 0)

                # zip and shuffle
                multibatch = list(zip(multi_input, multi_target))
                random.shuffle(multibatch)

                # iterate over batches in multibatch
                multibatch_parted = zip(*(iter(multibatch),) * batch_size)
                for part in multibatch_parted:
                    input, target = zip(*part)

                    # convert tuples of tensors into one tensor
                    input = torch.stack(input)
                    target = torch.stack(target)

                    loss, output = self.update_model(input, target, None)
                    input = input.detach()
                    target = target.detach()

                    # measure accuracy and record loss
                    acc = accumulate_acc(output, target, None, acc)
                    losses.update(loss, input.size(0))

                    batch_time.update(batch_timer.toc())  # measure elapsed time
                    data_timer.toc()

                    if ((self.config['base_print_freq'] > 0) and (i % self.config['base_print_freq'] == 0)) or (i + 1) == len(
                            train_loader):
                        self.log('[{0}/{1}]\t'
                                 '{batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                                 '{data_time.val:.4f} ({data_time.avg:.4f})\t'
                                 '{loss.val:.3f} ({loss.avg:.3f})\t'
                                 '{acc.val:.2f} ({acc.avg:.2f})'.format(
                            i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, acc=acc))

            train_accs.append(acc.avg)

            self.log(' * Train on {} original batches, Acc {acc.avg:.3f}'.format(len(train_loader), acc=acc))

            # Evaluate the performance of current task
            if val_loader != None:
                val_accs.append(self.validation(val_loader))

        print("All epochs ended")

    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        if self.valid_out_dim == 'ALL':
            self.valid_out_dim = 0  # Initialize it with zero
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())

    def save_model(self, filename):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', filename)
        torch.save(model_state, filename + '.pth')
        print('=> Save Done')

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'],
                                               output_device=self.config['gpuid'][0])
        return self


def accumulate_acc(output, target, task, meter):
    if 'All' in output.keys():  # Single-headed model
        meter.update(accuracy(output['All'], torch.argmax(target, 1)), len(torch.argmax(target, 1)))
    else:  # outputs from multi-headed (multi-task) model
        for t, t_out in output.items():
            inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
            if len(inds) > 0:
                t_out = t_out[inds]
                t_target = target[inds]
                meter.update(accuracy(torch.argamx(t_out, 1), torch.argmax(t_target, 1)), len(inds))

    return meter
