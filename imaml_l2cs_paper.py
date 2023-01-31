import clearml
import higher
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import torch.utils.model_zoo as model_zoo
from torchvision import transforms


from meta_dataloader import WETMetaLoader
from clear_training_utils import (parse_args,
                                  get_fc_params,
                                  get_non_ignored_params,
                                  get_ignored_params,
                                  getArch_weights,
                                  load_filtered_state_dict)

from meta_dataloader import WETMetaLoader

def set_params(model, param_vals):
        offset = 0
        for param in model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()

def regularization_loss(model, w_0, lam=0.0):
        """
        Add a regularization loss onto the weights
        The proximal term regularizes around the point w_0
        Strength of regularization is lambda
        lambda can either be scalar (type float) or ndarray (numpy.ndarray)
        """
        regu_loss = 0.0
        offset = 0
        regu_lam = lam # if type(lam) == float or np.float64 else utils.to_tensor(lam)
        if w_0.dtype == torch.float16:
            try:
                regu_lam = regu_lam.half()
            except:
                regu_lam = np.float16(regu_lam)
        for param in model.parameters():
            delta = param.view(-1) - w_0[offset:offset + param.nelement()].view(-1)
            if type(regu_lam) == float or np.float64:
                regu_loss += 0.5 * regu_lam * torch.sum(delta ** 2)
            else:
                # import ipdb; ipdb.set_trace()
                param_lam = regu_lam[offset:offset + param.nelement()].view(-1)
                param_delta = delta * param_lam
                regu_loss += 0.5 * torch.sum(param_delta ** 2)
            offset += param.nelement()
        return regu_loss

def to_numpy(x):
    if type(x) == np.ndarray:
        return x
    else:
        try:
            return x.data.numpy()
        except:
            return x.cpu().data.numpy()
def to_cuda(x):
    try:
        return x.cuda()
    except:
        return torch.from_numpy(x).float().cuda()
def to_cuda(x):
    try:
        return x.cuda()
    except:
        return torch.from_numpy(x).float().cuda()


def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")

def to_device(x, GPU=False):
    if GPU:
        return to_cuda(x)
    else:
        return to_tensor(x)

def matrix_evaluator(model, x, y_yb, y_pb, y_yc, y_pc, lam, regu_coef=1.0, lam_damping=10.0):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = to_device(lam, True)
        def evaluator(v):
            hvp = hessian_vector_product(
                    v, model=model, x=x, y_yb=y_yb, y_pb=y_pb,
                    y_yc=y_yc, y_pc=y_pc)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

def hessian_vector_product(vector, model, x, y_yb, y_pb, y_yc, y_pc, params=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if params is not None:
            offset = 0
            for param in model.parameters():
                param.data.copy_(params[offset:offset + param.nelement()].view(param.size()))
                offset += param.nelement()

        pitch, yaw = fast_net(x)

        s_loss_pitch_gaze = criterion(pitch, y_pb.to(torch.long))
        s_loss_yaw_gaze = criterion(yaw, y_yb.to(torch.long))
        pitch_predicted = softmax(pitch)
        yaw_predicted = softmax(yaw)
        pitch_predicted = \
        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
        yaw_predicted = \
        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

        s_loss_reg_pitch = reg_criterion(pitch_predicted, y_pc)
        s_loss_reg_yaw = reg_criterion(yaw_predicted, y_yc)

        # Total loss
        s_loss_pitch_gaze += s_loss_reg_pitch
        s_loss_yaw_gaze += s_loss_reg_yaw

        tloss = s_loss_pitch_gaze + s_loss_yaw_gaze

        grad_ft = torch.autograd.grad(tloss, model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        vec = to_device(vector, True)
        h = torch.sum(flat_grad * vec)
        hvp = torch.autograd.grad(h, model.parameters())
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat

def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """

    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose:
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x

def outer_step_with_grad(model, optimizer, grad, flat_grad=False):
    """
    Given the gradient, step with the outer optimizer using the gradient.
    Assumed that the gradient is a tuple/list of size compatible with model.parameters()
    If flat_grad, then the gradient is a flattened vector
    """
    check = 0
    for p in model.parameters():
        check = check + 1 if type(p.grad) == type(None) else check
    if check > 0:
    #     # initialize the grad fields properly
        print("POPSUTE")
        dummy_loss = regularization_loss(model,
                                         torch.cat([param.data.view(-1) for param in model.parameters()], 0).clone())
        dummy_loss.backward()  # this would initialize required variables
    if flat_grad:
        offset = 0
        grad = to_device(grad, True)
        for p in model.parameters():
            this_grad = grad[offset:offset + p.nelement()].view(p.size())
            p.grad.copy_(this_grad)
            offset += p.nelement()
    else:
        for i, p in enumerate(model.parameters()):
            p.grad = grad[i]
    optimizer.step()

if __name__ == '__main__':

    META_BATCH_SIZE = 1
    META_LR = 5e-5
    INNER_LOOP_LR = 1e-5
    EPOCHS = 30
    INNER_STEPS = 30
    SUPPORT_SIZE = 20
    QUERY_SIZE = 30

    LAMB = 30
    N_CG = 10
    CG_DAMPING = 1.0
    LAM_DAMPING = 10.0

    task = clearml.Task.init(project_name="meta", task_name="l2cs_imaml", tags="v2")
    logger = task.get_logger()
    parameters = task.connect({})
    parameters['meta_batch_size'] = META_BATCH_SIZE
    parameters['meta_lr'] = META_LR
    parameters['inner_loop_lr'] = INNER_LOOP_LR
    parameters['epochs'] = EPOCHS
    parameters['inner_steps'] = INNER_STEPS
    parameters['support_size'] = SUPPORT_SIZE
    parameters['query_size'] = QUERY_SIZE
    parameters['lamb'] = LAMB
    parameters['n_cg'] = N_CG
    parameters['cg_damping'] = CG_DAMPING
    parameters['lam_damping'] = LAM_DAMPING



    transformations = transforms.Compose([
        transforms.Resize(244),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])
    device = torch.device('cuda')

    softmax = nn.Softmax(dim=1).cuda(device)

    learner_net, pre_url = getArch_weights("ResNet18", 28)
    load_filtered_state_dict(learner_net, model_zoo.load_url(pre_url))
    # model = torch.load("/home/janek/software/L2CS-Net/models/meta/model_epoch14.pkl")
    learner_net = nn.DataParallel(learner_net)
    learner_net.to(device)

    fast_net, pre_url = getArch_weights("ResNet18", 28)
    load_filtered_state_dict(fast_net, model_zoo.load_url(pre_url))
    # model = torch.load("/home/janek/software/L2CS-Net/models/meta/model_epoch14.pkl")
    fast_net = nn.DataParallel(fast_net)
    fast_net.to(device)

    criterion = nn.CrossEntropyLoss().cuda(device)
    reg_criterion = nn.L1Loss().cuda(device)

    idx_tensor = [idx for idx in range(28)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(device)

    learner_inner_optimizer = torch.optim.SGD(learner_net.parameters(), INNER_LOOP_LR)
    learner_outer_optimizer = torch.optim.Adam(learner_net.parameters(), META_LR)
    fast_inner_optimizer = torch.optim.SGD(fast_net.parameters(), INNER_LOOP_LR)
    fast_outer_optimizer = torch.optim.Adam(fast_net.parameters(), META_LR)

    data = WETMetaLoader(annotations="/home/janek/software/L2CS-Net/meta_dataset_normalized/annotations.txt",
                         root="/home/janek/software/L2CS-Net/meta_dataset_normalized",
                         nshot_support=SUPPORT_SIZE, n_query=QUERY_SIZE,
                         transforms=transformations)

    meta_train = Subset(data, range(int(.8*len(data))))
    meta_test = Subset(data, range(int(.8*len(data)), len(data)))


    meta_train_loader = DataLoader(dataset=meta_train,
                                   batch_size=META_BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=False)

    meta_test_loader = DataLoader(dataset=meta_test,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=False)

    # meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=len(meta_train_loader), eta_min=0)

    lam = torch.tensor(LAMB).to(device)
    iters = 0
    for epoch in range(EPOCHS):

        for i, (b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb) in enumerate(meta_train_loader):
            print(f"Iter {i+1} / {len(meta_train_loader.dataset)}")

            qry_accs = []
            grad_list = []
            meta_grad = 0
            lam_grad = 0
            w_k = torch.cat([param.data.view(-1) for param in learner_net.parameters()], 0).clone()

            for s_im, s_lc, s_lb, q_im, q_lc, q_lb in zip(b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb):
                iters += 1
                set_params(fast_net, w_k.clone())
            # Prepare data
                s_im = torch.squeeze(s_im, 0)
                s_im = s_im.to(device)
                s_lc = torch.squeeze(s_lc, 0)
                s_lc = s_lc.to(device)
                s_lb = torch.squeeze(s_lb, 0)
                s_lb = s_lb.to(device)
                q_im = torch.squeeze(q_im, 0)
                q_im = q_im.to(device)
                q_lc = torch.squeeze(q_lc, 0)
                q_lc = q_lc.to(device)
                q_lb = torch.squeeze(q_lb, 0)
                q_lb = q_lb.to(device)

                s_label_pitch_gaze = Variable(s_lb[:, 0]).cuda(device)
                s_label_yaw_gaze = Variable(s_lb[:, 1]).cuda(device)
                s_label_pitch_cont_gaze = Variable(s_lc[:, 0]).cuda(device)
                s_label_yaw_cont_gaze = Variable(s_lc[:, 1]).cuda(device)

                q_label_pitch_gaze = Variable(q_lb[:, 0]).cuda(device)
                q_label_yaw_gaze = Variable(q_lb[:, 1]).cuda(device)
                q_label_pitch_cont_gaze = Variable(q_lc[:, 0]).cuda(device)
                q_label_yaw_cont_gaze = Variable(q_lc[:, 1]).cuda(device)

                # setup model
                # model.zero_grad()
                losses = []
                for step_number in range(INNER_STEPS):
                    fast_inner_optimizer.zero_grad()

                    pitch, yaw = fast_net(s_im)

                    s_loss_pitch_gaze = criterion(pitch, s_label_pitch_gaze.to(torch.long))
                    s_loss_yaw_gaze = criterion(yaw, s_label_pitch_gaze.to(torch.long))
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)
                    pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    s_loss_reg_pitch = reg_criterion(pitch_predicted, s_label_pitch_cont_gaze)
                    s_loss_reg_yaw = reg_criterion(yaw_predicted, s_label_yaw_cont_gaze)

                    # Total loss
                    s_loss_pitch_gaze += s_loss_reg_pitch
                    s_loss_yaw_gaze += s_loss_reg_yaw

                    loss_seq = s_loss_pitch_gaze + s_loss_yaw_gaze
                    fast_inner_optimizer.step()

                fast_inner_optimizer.zero_grad()
                regu_loss = regularization_loss(fast_net, w_k, lam)
                regu_loss.backward()
                fast_inner_optimizer.step()

                pitch_q, yaw_q = fast_net(q_im)

                q_loss_pitch_gaze = criterion(pitch_q, q_label_pitch_gaze.to(torch.long))
                q_loss_yaw_gaze = criterion(yaw_q, q_label_pitch_gaze.to(torch.long))
                pitch_predicted_q = softmax(pitch_q)
                yaw_predicted_q = softmax(yaw_q)
                pitch_predicted_q = \
                torch.sum(pitch_predicted_q * idx_tensor, 1) * 3 - 42
                yaw_predicted_q = \
                torch.sum(yaw_predicted_q * idx_tensor, 1) * 3 - 42

                q_loss_reg_pitch = reg_criterion(pitch_predicted_q, q_label_pitch_cont_gaze)
                q_loss_reg_yaw = reg_criterion(yaw_predicted_q, q_label_yaw_cont_gaze)

                # Total loss
                q_loss_pitch_gaze += q_loss_reg_pitch
                q_loss_yaw_gaze += q_loss_reg_yaw

                test_loss = q_loss_pitch_gaze + q_loss_yaw_gaze
                qry_accs.append(test_loss)
                test_grad = torch.autograd.grad(test_loss, fast_net.parameters())
                flat_grad = torch.cat([g.contiguous().view(-1) for g in test_grad])
                task_matrix_evaluator = matrix_evaluator(
                                            fast_net, s_im, s_label_yaw_gaze,
                                            s_label_pitch_gaze, s_label_yaw_cont_gaze,
                                            s_label_pitch_cont_gaze, lam)
                task_outer_grad = cg_solve(task_matrix_evaluator, flat_grad, N_CG, x_init=None)
                meta_grad += task_outer_grad / META_BATCH_SIZE

                lam_grad = 0.0
            outer_step_with_grad(learner_net, learner_outer_optimizer, meta_grad, True)
            print(sum(qry_accs) / len(qry_accs))
            logger.report_scalar("Query", "Loss", iteration=iters, value=(sum(qry_accs) / len(qry_accs)))
            # meta_scheduler.step()

        torch.save(learner_net.state_dict(), os.path.join("/home/janek/software/L2CS-Net/models/imaml_paper_l2cs", f"model_epoch{epoch}_state_dict.pkl"))
