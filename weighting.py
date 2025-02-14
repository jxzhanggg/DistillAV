import torch, sys, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger(__name__)

class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        # 获取多任务共享参数
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)
        # 全部共享参数的总维度

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        # 全部共享参数的的梯度拼接成1维向量
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            # self.task num 任务个数
            grads = torch.zeros(self.task_num, self.grad_dim).to(losses[0].device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    # 多任务，反向多次，但是有梯度累计啊
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                elif mode == 'autograd':
                    grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                    grads[tn] = torch.cat([g.view(-1) for g in grad])
                else:
                    raise ValueError('No support {} mode for gradient computation')
                # 共享参数的梯度置零
                self.zero_grad_share_params()
        else:
            if not isinstance(self.rep, dict):
                grads = torch.zeros(self.task_num, *self.rep.size()).to(losses[0].device)
            else:
                grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
            for tn, task in enumerate(self.task_name):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.

        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.

        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
                grads_sum_t = per_grads.reshape(self.task_num, self.rep.size()[0] * self.rep.size()[1], self.rep.size()[2]).sum(1)

            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
        
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None, other_loss = None):
        r"""This function is used to reset the gradients and make a backward.

        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                if other_loss is None:
                    self.rep.backward(transformed_grad)
                else:
                    self.rep.backward(transformed_grad, retain_graph=True)
                    other_loss.backward()
            else:
                assert False
                for tn, task in enumerate(self.task_name):
                    if other_loss is None:
                        rg = True if (tn+1)!=self.task_num else False
                    else:
                        rg = True
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)

                if not (other_loss is None):
                    other_loss.backward()
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from LibMTL.weighting.abstract_weighting import AbsWeighting


class MGDA(AbsWeighting):
    r"""Multiple Gradient Descent Algorithm (MGDA).
    
    This method is proposed in `Multi-Task Learning as Multi-Objective Optimization (NeurIPS 2018) <https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/isl-org/MultiObjectiveOptimization>`_. 

    Args:
        mgda_gn ({'none', 'l2', 'loss', 'loss+'}, default='none'): The type of gradient normalization.

    """
    def __init__(self):
        super(MGDA, self).__init__()
    
    def _find_min_norm_element(self, grads):

        def _min_norm_element_from2(v1v1, v1v2, v2v2):
            if v1v2 >= v1v1:
                gamma = 0.999
                cost = v1v1
                return gamma, cost
            if v1v2 >= v2v2:
                gamma = 0.001
                cost = v2v2
                return gamma, cost
            gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
            cost = v2v2 + gamma*(v1v2 - v2v2)
            return gamma, cost

        def _min_norm_2d(grad_mat):
            dmin = 1e8
            for i in range(grad_mat.size()[0]):
                for j in range(i+1, grad_mat.size()[0]):
                    c,d = _min_norm_element_from2(grad_mat[i,i], grad_mat[i,j], grad_mat[j,j])
                    if d < dmin:
                        dmin = d
                        sol = [(i,j),c,d]
            return sol

        def _projection2simplex(y):
            m = len(y)
            sorted_y = torch.sort(y, descending=True)[0]
            tmpsum = 0.0
            tmax_f = (torch.sum(y) - 1.0)/m
            for i in range(m-1):
                tmpsum+= sorted_y[i]
                tmax = (tmpsum - 1)/ (i+1.0)
                if tmax > sorted_y[i+1]:
                    tmax_f = tmax
                    break
            return torch.max(y - tmax_f, torch.zeros(m).to(y.device))

        def _next_point(cur_val, grad, n):
            proj_grad = grad - ( torch.sum(grad) / n )
            tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
            tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])

            skippers = torch.sum(tm1<1e-7) + torch.sum(tm2<1e-7)
            t = torch.ones(1).to(grad.device)
            if (tm1>1e-7).sum() > 0:
                t = torch.min(tm1[tm1>1e-7])
            if (tm2>1e-7).sum() > 0:
                t = torch.min(t, torch.min(tm2[tm2>1e-7]))

            next_point = proj_grad*t + cur_val
            next_point = _projection2simplex(next_point)
            return next_point

        MAX_ITER = 250
        STOP_CRIT = 1e-5
    
        grad_mat = grads.mm(grads.t())
        init_sol = _min_norm_2d(grad_mat)
        
        n = grads.size()[0]
        sol_vec = torch.zeros(n).to(grads.device)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec
    
        iter_count = 0

        while iter_count < MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = _next_point(sol_vec, grad_dir, n)

            v1v1 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*sol_vec.unsqueeze(0).repeat(n, 1)*grad_mat)
            v1v2 = torch.sum(sol_vec.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
            v2v2 = torch.sum(new_point.unsqueeze(1).repeat(1, n)*new_point.unsqueeze(0).repeat(n, 1)*grad_mat)
    
            nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < STOP_CRIT:
                return sol_vec
            sol_vec = new_sol_vec
            iter_count += 1
        return sol_vec
    
    def _gradient_normalizers(self, grads, loss_data, ntype):
        if ntype == 'l2':
            gn = grads.pow(2).sum(-1).sqrt()
        elif ntype == 'loss':
            gn = loss_data
        elif ntype == 'loss+':
            gn = loss_data * grads.pow(2).sum(-1).sqrt()
        elif ntype == 'none':
            gn = torch.ones_like(loss_data).to(loss_data.device)
        else:
            raise ValueError('No support normalization type {} for MGDA'.format(ntype))
        grads = grads / gn.unsqueeze(1).repeat(1, grads.size()[1])
        return grads
    
    def backward(self, losses, update_num, **kwargs):
       
        mgda_gn = "none"

        other_loss = losses[-1]
        losses = losses[:-1]

        grads = self._get_grads(losses, mode='backward')

        assert(self.rep_grad)
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        
        # avoid overflow in fp16 training added by jxzhang
        org_dtype = grads.dtype

        # print("shape:", grads.shape)
        # print(grads)
        # print("dot:",(grads[0] * grads[1]).sum())

        loss_data = torch.tensor([loss.item() for loss in losses]).to(losses[0].device)
        grads = self._gradient_normalizers(grads.float(), loss_data, ntype=mgda_gn) # l2, loss, loss+, none

        # print("shape:", grads.shape)
        # print(grads)
        # print("after norm dot:",(grads[0] * grads[1]).sum())

        sol = self._find_min_norm_element(grads)

        sol = sol.to(org_dtype)

        if self.rep_grad:
            self._backward_new_grads(sol, per_grads=per_grads, other_loss=other_loss)
        else:
            self._backward_new_grads(sol, grads=grads)
        
        # print("-" * 10, "backward", sol.detach().cpu().numpy(), '-' * 10)

        if update_num % 50 == 0:
            array =  sol.detach().cpu().numpy()
            logger.info(f"multi task weights: {array}.")

        return sol.detach().cpu().numpy()


class Aligned_MTL(AbsWeighting):
    r"""Aligned-MTL.
    
    This method is proposed in `Independent Component Alignment for Multi-Task Learning (CVPR 2023) <https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/SamsungLabs/MTL>`_. 

    """
    def __init__(self):
        super(Aligned_MTL, self).__init__()
        
    def backward(self, losses, update_num, **kwargs):
        

        other_loss = losses[-1]
        losses = losses[:-1]

        grads = self._get_grads(losses, mode='backward')

        assert(self.rep_grad)
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        

        org_dtype = grads.dtype
        grads = grads.float()
        
        M = torch.matmul(grads, grads.t()) # [num_tasks, num_tasks]
        lmbda, V = torch.symeig(M, eigenvectors=True)
        tol = (
            torch.max(lmbda)
            * max(M.shape[-2:])
            * torch.finfo().eps
        )
        rank = sum(lmbda > tol)

        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]

        sigma = torch.diag(1 / lmbda.sqrt())
        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())

        # print('B', B)
        alpha = B.sum(0)
        # alpha = B.sum(1) # ? 

        alpha = alpha.to(org_dtype)

        if self.rep_grad:
            self._backward_new_grads(alpha, per_grads=per_grads, other_loss=other_loss)
        else:
            self._backward_new_grads(alpha, grads=grads)
        

        if update_num % 50 == 0:
            array =  alpha.detach().cpu().numpy()
            logger.info(f"multi task weights: {array}.")
        
        return alpha.detach().cpu().numpy()