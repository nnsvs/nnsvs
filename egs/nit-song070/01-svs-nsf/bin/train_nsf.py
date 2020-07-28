# coding: utf-8
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import argparse
import os
from os.path import exists, join, splitext
import sys
import torch
import copy

from nnsvs.logger import getLogger
logger = None

@hydra.main(config_path="conf/train_nsf/config.yaml")
def my_app(config : DictConfig) -> None:
    global logger
    logger = getLogger(config.verbose)
    logger.info(config.pretty())

    assert config.nsf_root_dir
    nsf_root_dir = to_absolute_path(config.nsf_root_dir)
    sys.path.append(nsf_root_dir)
    import core_scripts.data_io.default_data_io as nii_dset
    import core_scripts.other_tools.list_tools as nii_list_tool
    import core_scripts.op_manager.op_manager as nii_op_wrapper
    import core_scripts.nn_manager.nn_manager as nii_nn_wrapper

    if config.nsf_type == "hn-sinc-nsf":
        sys.path.append(to_absolute_path(join(config.nsf_root_dir, "project/hn-sinc-nsf-9")))
    elif config.nsf_type == "hn-nsf":
        sys.path.append(to_absolute_path(join(config.nsf_root_dir, "project/hn-nsf")))
    elif config.nsf_type == "cyc-noise-nsf":
        sys.path.append(to_absolute_path(join(config.nsf_root_dir, "project/cyc-noise-nsf-4")))
    else:
        raise Exception(f"Unknown NSF type: {config.nsf_type}")

    import model as nsf_model
    
    # initialization
    # All NSF related settings are copied to argparse.Namespace object, because NSF core scripts are written
    # to work with argparse, not hydra.
    # Settings of file paths are converted to absolute ones(save_model_dir, trained_model, output_dir)
    args = argparse.Namespace()
    args.batch_size = config.nsf.args.batch_size
    args.epochs = config.nsf.args.epochs
    args.no_best_epochs = config.nsf.args.no_best_epochs
    args.lr = config.nsf.args.lr
    args.no_cuda = config.nsf.args.no_cuda
    args.seed = config.nsf.args.seed
    args.eval_mode_for_validation = config.nsf.args.eval_mode_for_validation
    args.model_forward_with_target = config.nsf.args.model_forward_with_target
    args.model_forward_with_file_name = config.nsf.args.model_forward_with_file_name
    args.shuffle = config.nsf.args.shuffle
    args.num_workers = config.nsf.args.num_workers
    args.multi_gpu_data_parallel = config.nsf.args.multi_gpu_data_parallel
    if config.nsf.args.save_model_dir != None:
        args.save_model_dir = to_absolute_path(config.nsf.args.save_model_dir)
    else:
        args.save_model_dir = None 
    args.not_save_each_epoch = config.nsf.args.not_save_each_epoch
    args.save_epoch_name = config.nsf.args.save_epoch_name
    args.save_trained_name = config.nsf.args.save_trained_name
    args.save_model_ext = config.nsf.args.save_model_ext
    if config.nsf.args.trained_model != None:
        args.trained_model = to_absolute_path(config.nsf.args.trained_model)
    else:
        args.trained_model = None
    args.ignore_training_history_in_trained_model = config.nsf.args.ignore_training_history_in_trained_model
    args.inference = config.nsf.args.inference
    # args.output_dir is set to config.nsf.model.test_output_dirs for inference stage
    if config.nsf.model.test_output_dirs != None:
        args.output_dir = to_absolute_path(config.nsf.model.test_output_dirs)
    else:
        args.output_dir=None
    args.optimizer = config.nsf.args.optimizer
    args.verbose = config.nsf.args.verbose
    
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if not args.inference:
        # prepare data io    
        params = {'batch_size':  args.batch_size,
                  'shuffle':  args.shuffle,
                  'num_workers': args.num_workers}

        # Load file list and create data loader
        train_list_path = to_absolute_path(config.data.train_no_dev.list_path)
        train_list = nii_list_tool.read_list_from_text(train_list_path)

        input_dirs = [to_absolute_path(x) for x in config.nsf.model.input_dirs]
        output_dirs = [to_absolute_path(x) for x in config.nsf.model.output_dirs]

        # If we pass config.nsf.model.input_* to NIIDataSet.f_calculate_stats(), 
        # it will overwrite them because it assumes them as List object, not "ListConfig" 
        # object(This mismatch results from the adaption of Hydra instead of simple config.py).
        # To avoid this, deepcopies of input_* are created.
        # FIXME
        input_exts = copy.deepcopy(config.nsf.model.input_exts)
        input_dims = copy.deepcopy(config.nsf.model.input_dims)
        input_reso = copy.deepcopy(config.nsf.model.input_reso)
        input_norm = copy.deepcopy(config.nsf.model.input_norm)

        train_set = nii_dset.NIIDataSetLoader("train_no_dev",
                                              train_list, 
                                              input_dirs,
                                              input_exts, 
                                              input_dims, 
                                              input_reso, 
                                              input_norm, 
                                              output_dirs,
                                              config.nsf.model.output_exts, 
                                              config.nsf.model.output_dims, 
                                              config.nsf.model.output_reso, 
                                              config.nsf.model.output_norm, 
                                              args.save_model_dir,
                                              params = params,
                                              truncate_seq = config.nsf.model.truncate_seq, 
                                              min_seq_len = config.nsf.model.minimum_len,
                                              save_mean_std = True,
                                              wav_samp_rate = config.nsf.model.wav_samp_rate)

        val_list_path = to_absolute_path(config.data.dev.list_path)
        val_list = nii_list_tool.read_list_from_text(val_list_path)

        # If we pass config.nsf.model.input_* to NIIDataSet.f_calculate_stats(), 
        # it will overwrite them because it assumes them as List object, not "ListConfig" 
        # object(This mismatch results from the adaption of Hydra instead of simple config.py).
        # To avoid this, deepcopies of input_* are created.
        # FIXME

        input_exts = copy.deepcopy(config.nsf.model.input_exts)
        input_dims = copy.deepcopy(config.nsf.model.input_dims)
        input_reso = copy.deepcopy(config.nsf.model.input_reso)
        input_norm = copy.deepcopy(config.nsf.model.input_norm)

        val_set = nii_dset.NIIDataSetLoader("dev",
                                            val_list,
                                            input_dirs,
                                            input_exts,
                                            input_dims,
                                            input_reso,
                                            input_norm,
                                            output_dirs,
                                            config.nsf.model.output_exts,
                                            config.nsf.model.output_dims,
                                            config.nsf.model.output_reso,
                                            config.nsf.model.output_norm,
                                            args.save_model_dir,
                                            params = params,
                                            truncate_seq= config.nsf.model.truncate_seq, 
                                            min_seq_len = config.nsf.model.minimum_len,
                                            save_mean_std = False,
                                            wav_samp_rate = config.nsf.model.wav_samp_rate)
        
        # Initialize the model and loss function
        model = nsf_model.Model(train_set.get_in_dim(),
                                train_set.get_out_dim(), 
                                args, train_set.get_data_mean_std())
        loss_wrapper = nsf_model.Loss(args)

        # initialize the optimizer
        optimizer_wrapper = nii_op_wrapper.OptimizerWrapper(model, args)

        # if necessary, resume training
        if not args.trained_model:
            checkpoint = None 
        else:
            checkpoint = torch.load(args.trained_model)
            
            # start training
        logger.info(f"Start {config.nsf_type} training. This may take several days.")
        nii_nn_wrapper.f_train_wrapper(args, model, 
                                       loss_wrapper, device,
                                       optimizer_wrapper,
                                       train_set, val_set, checkpoint)
    else:
        # for inference

        # default, no truncating, no shuffling
        params = {'batch_size':  args.batch_size,
                  'shuffle':  False,
                  'num_workers': args.num_workers}
        test_list_path = to_absolute_path(config.data.eval.list_path)
        test_list = nii_list_tool.read_list_from_text(test_list_path)

        # If we pass config.nsf.model.input_* to NIIDataSet.f_calculate_stats(), 
        # it will overwrite them because it assumes them as List object, not "ListConfig" 
        # object(This mismatch results from the adaption of Hydra instead of simple config.py).
        # To avoid this, deepcopies of input_* are created.
        # FIXME
        input_exts = copy.deepcopy(config.nsf.model.input_exts)
        input_dims = copy.deepcopy(config.nsf.model.input_dims)
        input_reso = copy.deepcopy(config.nsf.model.input_reso)
        input_norm = copy.deepcopy(config.nsf.model.input_norm)

        test_input_dirs = [to_absolute_path(x) for x in config.nsf.model.test_input_dirs]

        test_set = nii_dset.NIIDataSetLoader("eval",
                                             test_list, 
                                             test_input_dirs,
                                             input_exts, 
                                             input_dims, 
                                             input_reso, 
                                             input_norm, 
                                             [],
                                             config.nsf.model.output_exts, 
                                             config.nsf.model.output_dims, 
                                             config.nsf.model.output_reso, 
                                             config.nsf.model.output_norm, 
                                             args.save_model_dir, 
                                             params = params,
                                             truncate_seq = None,
                                             min_seq_len = None,
                                             save_mean_std = False,
                                             wav_samp_rate = config.nsf.model.wav_samp_rate)
        # Initialize the model
        model = nsf_model.Model(test_set.get_in_dim(),
                                test_set.get_out_dim(), 
                                args)

        if not args.trained_model:
            print("trained_model is not set, so try to load default trained model")
            default_trained_model_path = join(args.save_model_dir,
                                              "{}{}".format(args.save_trained_name,
                                                            args.save_model_ext))
            if not exists(default_trained_model_path):
                raise Exception("No trained model found")
            checkpoint = torch.load(default_trained_model_path)
        else:
            checkpoint = torch.load(args.trained_model)
        # do inference and output data
        nii_nn_wrapper.f_inference_wrapper(args, model, device,
                                           test_set, checkpoint)

def entry():
    my_app()

if __name__ == "__main__":
    my_app()
                            
