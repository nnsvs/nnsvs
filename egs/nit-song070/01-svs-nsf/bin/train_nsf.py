# coding: utf-8
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

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
    torch.manual_seed(config.nsf.args.seed)
    use_cuda = not config.nsf.args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # fix config.nsf.args.save_model_dir 
    logger.info(f"config.nsf.args.save_model_dir is converted to absolute path by NNSVS.")
    config.nsf.args.save_model_dir = to_absolute_path(config.nsf.args.save_model_dir)
    
    if not config.nsf.args.inference:
        # prepare data io    
        params = {'batch_size':  config.nsf.args.batch_size,
                  'shuffle':  config.nsf.args.shuffle,
                  'num_workers': config.nsf.args.num_workers}

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
                                              config.nsf.args.save_model_dir,
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
                                            config.nsf.args.save_model_dir,
                                            params = params,
                                            truncate_seq= config.nsf.model.truncate_seq, 
                                            min_seq_len = config.nsf.model.minimum_len,
                                            save_mean_std = False,
                                            wav_samp_rate = config.nsf.model.wav_samp_rate)
        
        # Initialize the model and loss function
        # Originally nsf_mode.Model requires args as Namespace Object, not DictConfig object.
        # But hydra uses ArgumentParser internally so we can't use nii_arg_parse.f_args_parsed().
        # Ugly duck typing :(
        model = nsf_model.Model(train_set.get_in_dim(),
                                train_set.get_out_dim(), 
                                config.nsf.args, train_set.get_data_mean_std())
        loss_wrapper = nsf_model.Loss(config.nsf.args)

        # initialize the optimizer
        optimizer_wrapper = nii_op_wrapper.OptimizerWrapper(model, config.nsf.args)

        # if necessary, resume training
        if not config.nsf.args.trained_model:
            checkpoint = None 
        else:
            checkpoint = torch.load(to_absolute_path(config.nsf.args.trained_model))
            
            # start training
        logger.info(f"Start {config.nsf_type} training. This may take several days.")
        nii_nn_wrapper.f_train_wrapper(config.nsf.args, model, 
                                       loss_wrapper, device,
                                       optimizer_wrapper,
                                       train_set, val_set, checkpoint)
    else:
        # for inference

        # default, no truncating, no shuffling
        params = {'batch_size':  config.nsf.args.batch_size,
                  'shuffle':  False,
                  'num_workers': config.nsf.args.num_workers}
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
        # Overwrite output_dir setting
        logger.info(f"NSF setting of config.nsf.args.output_dir is overwritten with {config.nsf.model.test_output_dirs} by NNSVS.")
        config.nsf.args.output_dir = to_absolute_path(config.nsf.model.test_output_dirs)

        save_model_dir = to_absolute_path(config.nsf.args.save_model_dir)

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
                                             save_model_dir, 
                                             params = params,
                                             truncate_seq = None,
                                             min_seq_len = None,
                                             save_mean_std = False,
                                             wav_samp_rate = config.nsf.model.wav_samp_rate)
        # Initialize the model
        # Originally nsf_mode.Model requires args as Namespace Object, not DictConfig object.
        # But hydra uses ArgumentParser internally so we can't use nii_arg_parse.f_args_parsed().
        # Ugly duck typing :(
        model = nsf_model.Model(test_set.get_in_dim(),
                                test_set.get_out_dim(), 
                                config.nsf.args)

        if not config.nsf.args.trained_model:
            print("config.nsf.args.trained_model is not set, so try to load default trained model")
            default_trained_model_path = to_absolute_path(join(config.nsf.args.save_model_dir,
                                                               "{}{}".format(config.nsf.args.save_trained_name,
                                                                             config.nsf.args.save_model_ext)))
            if not exists(default_trained_model_path):
                raise Exception("No trained model found")
            checkpoint = torch.load(default_trained_model_path)
        else:
            checkpoint = torch.load(to_absolute_path(config.nsf.args.trained_model))
        # do inference and output data
        nii_nn_wrapper.f_inference_wrapper(config.nsf.args, model, device,
                                           test_set, checkpoint)

def entry():
    my_app()

if __name__ == "__main__":
    my_app()
                            
