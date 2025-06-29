# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

from pathlib import Path
import pprint
import yaml

from dap_behavior.jepa.src.utils.distributed import init_distributed

from dap_behavior.jepa.evals.scaffold import main as eval_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--ckpt', type=str, default=None,
    help='path to checkpoint to load')


def process_main(rank, fname, world_size, devices, pretrain_folder=None, pretrain_checkpoint=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        if pretrain_folder is not None:
            params['pretrain']['folder'] = pretrain_folder
            params['pretrain']['checkpoint'] = pretrain_checkpoint
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the eval with loaded config
    eval_main(params['eval_name'], args_eval=params)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.ckpt is not None:
        pretrain_folder = str(Path(args.ckpt).parent)
        pretrain_checkpoint = str(Path(args.ckpt).name)
    else:
        pretrain_folder = pretrain_checkpoint = None
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices),
            kwargs=dict(pretrain_folder=pretrain_folder, pretrain_checkpoint=pretrain_checkpoint)
        ).start()
