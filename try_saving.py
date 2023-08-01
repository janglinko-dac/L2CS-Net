import torch

model = torch.load("/home/janek/software/L2CS-Net/models/l2cs_maml_gc/model_epoch92.pkl")
model = torch.nn.DataParallel(model, device_ids=[0])
model.train()
model.to("cuda")

torch.save(model.state_dict(), "/home/janek/software/L2CS-Net/models/l2cs_maml_gc/model_epoch92_state_dict.pkl")