#-------------------------------------#
# Train on the dataset
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (Loss, ModelEMA, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_classes, seed_everything,
                         show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
When training your own target detection model, you must pay attention to the following points:
1. Before training, carefully check whether your format meets the requirements. The library requires the data set format to be in VOC format, and the required content includes input images and labels.
   The input image is a .jpg image. There is no need to fix the size. It will be automatically resized before being passed in for training.
   Grayscale images will be automatically converted into RGB images for training, and there is no need to modify them yourself.
   If the suffix of the input image is not jpg, you need to batch convert it to jpg before starting training.

   The label is in .xml format, and there will be target information that needs to be detected in the file. The label file corresponds to the input image file.

2. The size of the loss value is used to judge whether it converges. What is more important is that there is a convergence trend, that is, the loss of the verification set continues to decrease. If the loss of the verification set basically does not change, the model has basically converged.
   The specific size of the loss value has no meaning. Large or small only depends on the calculation method of the loss, not close to 0. If you want to make the loss look better, you can directly go to the corresponding loss function and divide it by 10,000.
   The loss values ​​during the training process will be saved in the loss_%Y_%m_%d_%H_%M_%S folder under the logs folder.
   
3. The trained weight files are saved in the logs folder. Each training generation (Epoch) contains several training steps (Step), and each training step (Step) performs a gradient descent.
   If you only train a few steps, they will not be saved. The concepts of Epoch and Step need to be clarified.
'''
if __name__ == "__main__":
    #---------------------------------#
    # Cuda Whether to use Cuda
    # No GPU can be set to False
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    # Seed is used to fix the random seed
    # So that each independent training can obtain the same results
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    # distributed is used to specify whether to use a single machine with multiple cards for distributed operation.
    # Terminal commands are only supported on Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
    # Under Windows systems, DP mode is used by default to call all graphics cards, and DDP is not supported.
    # DP mode:
    # Set distributed = False
    # Enter CUDA_VISIBLE_DEVICES=0,1 in the terminal python train.py
    # DDP mode:
    # Set distributed = True
    # Enter CUDA_VISIBLE_DEVICES=0,1 in the terminal python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    # sync_bn Whether to use sync_bn, multiple cards are available in DDP mode
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    # fp16 whether to use mixed precision training
    # Can reduce video memory by about half, requires pytorch1.7.1 or above
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    # classes_path points to the txt under model_data, which is related to your own training data set
    # Before training, be sure to modify classes_path so that it corresponds to your own data set.
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------------------------------------------------------------------------------#
    # Since there are no pre-trained weights, we will start training from 0
    # Set model_path = '', Freeze_Train = False. At this time, training starts from 0 and there is no process of freezing the trunk.
    # Since the dataset is smaller, we will set a larger training generation and appropriate batch_size
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = ''
    #------------------------------------------------------#
    # input_shape The input shape size must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #------------------------------------------------------#
    # The version of yolov8 used by phi
    # n: corresponds to yolov8_n
    # s: corresponds to yolov8_s
    # m: corresponds to yolov8_m
    # l: corresponds to yolov8_l
    # x: corresponds to yolov8_x
    #------------------------------------------------------#
    phi             = 's'
    #----------------------------------------------------------------------------------------------------------------------------#
    # pretrained Whether to use the pre-trained weights of the backbone network. The weights of the backbone are used here, so they are loaded when the model is built.
    # If model_path is set, the weights of the backbone do not need to be loaded, and the pretrained value is meaningless.
    # If model_path is not set, pretrained = True, only the backbone is loaded to start training.
    # If model_path is not set, pretrained = False, Freeze_Train = False, training starts from 0 and there is no process of freezing the trunk.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    # mosaic mosaic data enhancement.
    # mosaic_prob The probability of using mosaic data enhancement for each step. The default is 50%.
    #
    # mixup Whether to use mixup data enhancement, only valid when mosaic=True.
    # Only the mosaic-enhanced images will be mixed up.
    # mixup_prob The probability of using mixup data enhancement after mosaic, default 50%.
    # The total mixup probability is mosaic_prob * mixup_prob.
    #
    # special_aug_ratio refers to YoloX, because the training images generated by Mosaic are far away from the real distribution of natural images.
    # When mosaic=True, this code will enable mosaic in the special_aug_ratio range.
    # The default is the first 70% of epochs, and 100 generations will open 70 generations.
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    # label_smoothing label smoothing. Generally below 0.01. Such as 0.01, 0.005.
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #----------------------------------------------------------------------------------------------------------------------------#
    # Training is divided into two phases, the freezing phase and the defrosting phase. The freezing stage is set up to meet the training needs of students with insufficient machine performance.
    # Freeze training requires less video memory and the graphics card is very poor. You can set Freeze_Epoch equal to UnFreeze_Epoch and Freeze_Train = True. At this time, only freeze training is performed.
    #      
    # Here are some parameter setting suggestions for trainers to flexibly adjust according to their own needs:
    # (1) Start training from the pre-trained weights of the entire model:
    #       Adam:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (not frozen)
    #       SGD:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (not frozen)
    # Among them: UnFreeze_Epoch can be adjusted between 100-300.
    # (2) Training from 0:
    # Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (do not freeze training)
    # Among them: UnFreeze_Epoch should not be less than 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    # (3) Batch_size setting:
    # Within the range that the graphics card can accept, the larger is better. Insufficient video memory has nothing to do with the size of the data set. If it prompts insufficient video memory (OOM or CUDA out of memory), please adjust the batch_size smaller.
    # Affected by the BatchNorm layer, the minimum batch_size is 2 and cannot be 1.
    # Under normal circumstances, Freeze_batch_size is recommended to be 1-2 times of Unfreeze_batch_size. It is not recommended that the setting gap is too large, because it is related to the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Freeze phase training parameters
    # At this time, the backbone of the model is frozen, and the feature extraction network does not change.
    # It occupies a small amount of video memory and only fine-tunes the network.
    # Init_Epoch is the current training generation of the model. Its value can be greater than Freeze_Epoch, such as setting:
    #                       Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    # The freezing phase will be skipped, starting directly from generation 60, and adjusting the corresponding learning rate.
    # (Used when continuing practice from a break point)
    # Freeze_Epoch Freeze_Epoch of model freeze training
    # (Invalid when Freeze_Train=False)
    # Freeze_batch_size model freeze training batch_size
    # (Invalid when Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 4
    #------------------------------------------------------------------#
    # Training parameters during the unfreezing phase
    # At this time, the backbone of the model is no longer frozen, and the feature extraction network will change.
    # It occupies a large amount of video memory and all network parameters will change.
    # UnFreeze_Epoch The total number of epochs trained by the model
    # SGD takes longer to converge, so set a larger UnFreeze_Epoch
    # Adam can use a relatively small UnFreeze_Epoch
    # Unfreeze_batch_size batch_size of the model after unfreezing
    # Since the data set is small, set a smaller batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 2
    #------------------------------------------------------------------#
    # Freeze_Train whether to perform freeze training
    # Since the data set is small and training starts from 0, frozen training is not performed
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    # Other training parameters: learning rate, optimizer, learning rate decrease related
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Init_lr The maximum learning rate of the model
    # Min_lr The minimum learning rate of the model, the default is 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type The type of optimizer used. Optional options include adam and sgd.
    # When using the Adam optimizer it is recommended to set Init_lr=1e-3
    # When using the SGD optimizer it is recommended to set Init_lr=1e-2
    # momentum The momentum parameter used internally by the optimizer
    # weight_decay weight decay to prevent overfitting
    # adam will cause weight_decay error, it is recommended to set it to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    # The learning rate decrease method used by lr_decay_type, the options are step and cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    # save_period how many epochs to save the weight once
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    # save_dir is the folder where weights and log files are saved.
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    # eval_flag Whether to evaluate during training, the evaluation object is the verification set
    # After installing the pycocotools library, the evaluation experience will be better.
    # eval_period represents how many epochs to evaluate once. Frequent evaluation is not recommended.
    # Evaluation takes a lot of time, and frequent evaluation will cause training to be very slow.
    # The mAP obtained here will be different from that obtained by get_map.py for two reasons:
    # (1) The mAP obtained here is the mAP of the verification set.
    # (2) The evaluation parameters set here are conservative in order to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    # num_workers is used to set whether to use multi-threading to read data
    # Turning it on will speed up data reading, but will take up more memory.
    # Computers with smaller memory can be set to 2 or 0
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    # train_annotation_path training image path and label
    # val_annotation_path validates image paths and labels
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    seed_everything(seed)
    #------------------------------------------------------#
    # Set the graphics card used
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #------------------------------------------------------#
    # Get classes and anchor
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #----------------------------------------------------#
    # Download pre-trained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)
            
    #------------------------------------------------------#
    # Create yolo model
    #------------------------------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != '':
        #------------------------------------------------------#
        # For the weight file, please see README and download it from Baidu Netdisk.
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        # Load according to the Key of the pre-trained weights and the Key of the model
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        # Shows no matching Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "...\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "...\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mWarm reminder, it is normal that the head part is not loaded, and it is an error that the Backbone part is not loaded.\033[0m")

    #----------------------#
    # Get the loss function
    #----------------------#
    yolo_loss = Loss(model)
    #----------------------#
    # Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    # Torch 1.2 does not support amp. It is recommended to use torch 1.7.1 and above to use fp16 correctly.
    # Therefore, torch1.2 shows "could not be resolve" here
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    # Multi-SIM synchronization Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            # Multi-card parallel operation
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            
    #----------------------------#
    # weight smoothing
    #----------------------------#
    ema = ModelEMA(model_train)
    
    #---------------------------#
    # Read the txt corresponding to the data set
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        # The total training generation refers to the total number of times the entire data is traversed
        # The total training step size refers to the total number of times of gradient descent
        # Each training generation contains several training steps, and gradient descent is performed once at each training step.
        # Only the minimum training generation is recommended here, there is no upper limit, and only the defrosting part is considered in the calculation.
        #----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('The data set is too small to train, please expand the data set.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training step size to above %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] The total amount of training data in this run is %d, Unfreeze_batch_size is %d, a total of %d Epochs are trained, and the total training step is calculated to be %d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since the total training step size is %d, which is less than the recommended total step size %d, it is recommended to set the total generation to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    # Backbone feature extraction network features are universal, and frozen training can speed up training.
    # It can also prevent the weights from being destroyed in the early stages of training.
    # Init_Epoch is the starting generation
    # Freeze_Epoch is the generation that freezes training
    # UnFreeze_Epoch total training generation
    # If it prompts OOM or insufficient video memory, please adjust the Batch_size smaller.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        # Freeze certain portions of training
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        # If you do not freeze training, directly set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        # Determine the current batch_size and adaptively adjust the learning rate
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        # Select optimizer based on optimizer_type
        #---------------------------------------#
        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        #---------------------------------------#
        # Obtain the formula for learning rate decrease
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        # Determine the length of each generation
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0:
            raise ValueError("The data set is too small to continue training, please expand the data set.")
        
        # If the validation set stride is 0, set it to 1 to allow training on a small validation set
        if epoch_step_val == 0:
            epoch_step_val = 1

        if ema:
            ema.updates     = epoch_step * Init_Epoch
        
        #---------------------------------------#
        # Build a dataset loader.
        #---------------------------------------#
        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        # Record the map curve of eval
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        # Start model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            # If the model has a frozen learning part
            # Then unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                # Determine the current batch_size and adaptively adjust the learning rate
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                # Obtain the formula for learning rate decrease
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0:
                    raise ValueError("The data set is too small to continue training, please expand the data set.")
                
                # If the validation set stride is 0, set it to 1 to allow training on a small validation set
                if epoch_step_val == 0:
                    epoch_step_val = 1
                    
                if ema:
                    ema.updates     = epoch_step * epoch

                if distributed:
                    batch_size  = batch_size // ngpus_per_node
                    
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag   = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
            
            # After training is completed, confusion matrix heat map and feature visualization map are generated.
            print("\nTraining is completed, confusion matrix heat map and feature visualization map are generated...")
            import subprocess
            subprocess.run(["python", "generate_plots.py"], cwd=".")
            print("Confusion matrix heat map and feature visualization map have been generated")
