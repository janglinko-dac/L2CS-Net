import torch

model = torch.load("/home/janek/software/L2CS-Net/models/meta/model_epoch29.pkl")
model = torch.nn.DataParallel(model, device_ids=[0])
model.train()
model.to("cuda")

torch.save(model.state_dict(), "/home/janek/software/L2CS-Net/models/meta/model_epoch29_state_dict2.pkl")