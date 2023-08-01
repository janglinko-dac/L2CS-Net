from clear_training_utils import (parse_args,
                                  get_fc_params,
                                  get_non_ignored_params,
                                  get_ignored_params,
                                  getArch_weights,
                                  load_filtered_state_dict)
from meta_training_hydranet import Hydrant
import torch

model, pre_url = getArch_weights("ResNet18", 28)
model = torch.nn.DataParallel(model, device_ids=[0])
state_dict = torch.load("/home/janek/software/L2CS-Net/models/l2cs_maml_gc/model_epoch92_state_dict.pkl")

model.load_state_dict(state_dict)

# model = Hydrant("efficientnet_b0")
# state_dict = torch.load("/home/janek/software/L2CS-Net/models/hydrant_effb3/model_epoch23_state_dict.pkl")
# model.load_state_dict(state_dict)