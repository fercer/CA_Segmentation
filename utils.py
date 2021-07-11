import os
import shutil
import torch


def create_exp_dir(path, scripts_to_save=None):

  print('Experiment dir : {}'.format(path))
