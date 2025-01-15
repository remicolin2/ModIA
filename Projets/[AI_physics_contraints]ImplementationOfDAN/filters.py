"""
This file contains the DAN and function to construct the neural networks
"""
import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal as Mvn
import numpy as np 
import math 
import manage_exp
import lin2d_exp
import time 
class DAN(nn.Module):
    """
    A Data Assimilation Network class
    """
    def __init__(self, a_kwargs, b_kwargs, c_kwargs):

        nn.Module.__init__(self)
        self.a = ConstructorA(**a_kwargs)
        self.b = ConstructorB(**b_kwargs)
        self.c = ConstructorC(**c_kwargs)
        self.scores = {
            "RMSE_b": [],
            "RMSE_a": [],
            "LOGPDF_b": [],
            "LOGPDF_a": [],
            "LOSS": []}

    def forward(self, ha, x, y):
        """
        forward pass in the DAN
        """

        # TODO 2.2
        # propagate past mem into prior mem
        prior_state = self.b(ha)
        # translate prior mem into prior pdf        
        pdf_b = self.c(prior_state)
        # analyze prior mem
        posterior_state = self.a(torch.cat((prior_state, y), dim=1))
        # translate post mem into post pdf
        pdf_a = self.c(posterior_state)
    
        logpdf_a = -torch.mean(pdf_a.log_prob(x))
        logpdf_b = -torch.mean(pdf_b.log_prob(x))
        
        # TODO 2.4 rewrite loss 
        loss = logpdf_a+logpdf_b
        
        # Compute scores
        with torch.no_grad():
            if logpdf_a is not None:
                self.scores["RMSE_b"].append(torch.mean(torch.norm(
                    pdf_b.mean - x, dim=1)*x.size(1)**-.5).item())
                self.scores["RMSE_a"].append(torch.mean(torch.norm(
                    pdf_a.mean - x, dim=1)*x.size(1)**-.5).item())
                self.scores["LOGPDF_b"].append(logpdf_b.item())
                self.scores["LOGPDF_a"].append(logpdf_a.item())
                self.scores["LOSS"].append(loss.item())
                
        return loss, posterior_state

    def clear_scores(self):
        """ clear the score lists
        """
        for v in self.scores.values():
            v.clear()

class Id(nn.Module):
    """ A simple id function
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        """ trivial
        """
        return x

class Cst(nn.Module):
    """ A constant scale_vec
    """
    def __init__(self, init, dim=None):
        nn.Module.__init__(self)
        if isinstance(init, torch.Tensor):
            self.c = init.unsqueeze(0)
        else:
            raise NameError("Cst init unknown")

    def forward(self, x):
        return self.c.expand(x.size(0), self.c.size(0))

class Lin2d(nn.Module):
    # rotation dymnamics
    def __init__(self, x_dim, N, dt, init,
                 window=None):
        assert(x_dim == 2)
        nn.Module.__init__(self)        
        theta = np.pi / 100        
        linM = torch.Tensor([[np.cos(theta),np.sin(theta)],
                             [-np.sin(theta),np.cos(theta)]])
        self.linM = linM
        self.linMt = linM.t()
        self.x_dim = x_dim
        self.N = N
        
    def forward(self, x):
        for _ in range(self.N):            
            x = torch.matmul(x, self.linMt)
        return x


class EDO(nn.Module):
    """ Integrates an EDO with RK4
    """
    def __init__(self, x_dim, N, dt, init,
                 window=None):
        nn.Module.__init__(self)
        self.x_dim = x_dim
        self.N = N
        self.dt = dt
        if init == "95":
            """ Lorenz95 initialization
            """
            self.window = (-2, -1, 0, 1)
            self.diameter = 4
            self.A = torch.tensor([[[0., 0., 0., 0.],
                                  [-1., 0., 0., 0.],
                                  [0., 0., 0., 0.],
                                  [0., 1., 0., 0.]]])
            self.b = torch.tensor([[0., 0., -1., 0.]])
            self.c = torch.tensor([8.])
        else:
            raise NameError("EDO init not available")

    def edo(self, x):
        """v=
        x-2 x-1 x0 x1
        |   |   |  |
        """
        v = torch.cat(
            [torch.roll(x.unsqueeze(1), -i, 2) for i in self.window], 1)
        v = torch.transpose(v, 1, 2)
        v_flat = v.reshape(-1, self.diameter)
        dx = torch.nn.functional.bilinear(v_flat, v_flat, self.A)\
            + torch.nn.functional.linear(v_flat, self.b, self.c)
        return dx.view(x.size(0), x.size(1))

    def forward(self, x):
        for _ in range(self.N):
            k1 = self.edo(x)
            k2 = self.edo(x + 0.5*self.dt*k1)
            k3 = self.edo(x + 0.5*self.dt*k2)
            k4 = self.edo(x + self.dt*k3)
            x = x + (self.dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        return x

class FullyConnected(nn.Module):
    """ Fully connected NN ending with a linear layer
    """
    def __init__(self, layers, activation_classname):
        nn.Module.__init__(self)
        n = len(layers)
        self.lins = nn.ModuleList(
            [nn.Linear(d0, d1) for
             d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList(
            [eval(activation_classname)() for _ in range(n-2)])

    def forward(self, h):
        for lin, act in zip(self.lins[:-1], self.acts):
            h = act(lin(h))
        return self.lins[-1](h)

class FcZero(nn.Module):
    """
    Fully connected neural network with ReZero trick
    """
    def __init__(self, dim, deep, activation_classname):
        """
        layers: the list of the layers dimensions
        """
        # TODO 2.1 correct an error
        nn.Module.__init__(self)
        layers = (deep+1)*[dim]
        self.lins = nn.ModuleList(
            [nn.Linear(d0, d1) for
             d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList(
            [eval(activation_classname)() for _ in range(deep)])
        self.alphas = nn.Parameter(torch.zeros(deep), requires_grad=True)

    def forward(self, h):
        for (lin, act, alpha) in zip(self.lins, self.acts, self.alphas):
            h = h + alpha*act(lin(h))
        return h        

class FcZeroLin(nn.Module):
    """
    FcZero network ending with linear layer
    """
    def __init__(self, in_dim, out_dim, deep, activation_classname):
        """
        layers: the list of the layers dimensions
        """
        nn.Module.__init__(self)
        self.fcZero = FcZero(in_dim, deep, activation_classname)
        self.out_dim = out_dim
        assert(out_dim <= in_dim)
        self.lin = FullyConnected([in_dim, out_dim], activation_classname)

    def forward(self, h):
        h = self.fcZero(h)
        h = self.lin(h)
        return h
    


class GaussianDiag(Mvn):
    """
    Return a pytorch Gaussian pdf with diag covariance
    """
    def __init__(self, x_dim, vec):
        """args is a (x_dim, vec)
        vec shape is (batch size, xdim + 1)
        mu is the first x_dim coeff of vec
        c is the last coeff of vec
        scale_tril = e^c*I
        return Normal(mu,e^c*I)
        """
        vec_dim = vec.size(-1)
        assert(vec_dim == x_dim + 1)
        loc = vec[:, :x_dim]
        scale_tril = torch.eye(x_dim)\
                          .unsqueeze(0)\
                          .expand(vec.size(0), -1, -1)
        scale_tril = torch.exp(vec[:, x_dim])\
                          .view(vec.size(0), 1, 1)*scale_tril           
        Mvn.__init__(self, loc=loc, scale_tril=scale_tril)


class Gaussian(nn.Module):
    # Return a set of Gaussian pdf Normal( mu_i, Lambda_i * Lambda_i')
    # where i is an index of batch size
    # convert vec to the mean mu and the lower-triangular matrix Lambda
    def __init__(self, x_dim, vec, inds):
        """args is a (x_dim, vec)
        loc is the first x_dim coeff of vec         
        scale_tril is filled diagonal by diagonal, starting by the main one
        (which is exponentiated to ensure strict positivity)
        """
        vec_dim = vec.size(-1)
        bs = vec.size(0) # batch size
        # get mu
        loc = vec[:, :x_dim]
        # build Lambda
        diaga = vec[:, x_dim:2*x_dim] # non-exp diag terms
        #print('diaga',diaga.shape)
        diagoff = vec[:, 2*x_dim:None]
        #print('diagoff',diagoff.shape)
        # TODO 1.2
        # modify diaga to make it numerically stable
        # using minexp and maxexp
        minexp = torch.Tensor([-8.0])
        maxexp = torch.Tensor([8.0])        
        diaga = torch.clamp(diaga, min=minexp, max = maxexp)
        lbda = torch.cat((torch.exp(diaga), diagoff), 1)  # take exp on diaga
        scale_tril = torch.zeros(vec.size(0), x_dim, x_dim)
        scale_tril[:, inds[0], inds[1]] = lbda
        
        # INIT gaussian module
        # TODO 1.5 make the code more efficient
        
        self.mu = loc
        self.Lambda = scale_tril
        self.mean = self.mu
        self.variance = torch.exp(2 * diaga)
        self.covariance_matrix = torch.zeros(bs, x_dim, x_dim)
        self.cov_inv_ = torch.zeros(bs, x_dim, x_dim)
        self.cov_log_det_ = torch.zeros(bs, 1)
        logcst = np.log(2 * np.pi) / 2

        # Version plus rapide du code 
        start = time.time()
        
        cov = torch.bmm(self.Lambda, self.Lambda.transpose(1, 2))
        cov_log_det = torch.sum(logcst + diaga, dim=1)
        self.covariance_matrix = cov
        self.cov_inv_ = 0
        self.cov_log_det_ = cov_log_det.unsqueeze(1)
        #print("temps code amélioré TODO 1.5 : ", end - start)
        """
        for i in range(bs):
            cov = torch.mm(self.Lambda[i,:,:],self.Lambda[i,:,:].T)
            cov_inv = torch.cholesky_inverse(self.Lambda[i,:,:])
            cov_log_det = torch.sum( logcst + diaga[i,:] )
            self.covariance_matrix[i,:,:] = cov
            self.cov_inv_[i,:,:] = cov_inv
            self.cov_log_det_[i,0] = cov_log_det
        #end = time.time()
        #print("temps code normal TODO 1.5 : ", end - start)"""
    def log_prob(self, x):
        # TODO 1.2 compute logprob using self.mu and self.Lambda
        # Rewrite the term_1
        # x shape is (batch size, dim of x)
        # return log probability of x on the normal distribution
        # shape: (batch size, 1)
        bs = x.shape[0]
        logprob = torch.zeros(bs,1)
        mean_diff = x - self.mu
        L = self.Lambda
        z, _ = torch.triangular_solve(mean_diff.unsqueeze(-1), L, upper=False)

        term_1 = -0.5 * torch.bmm(z.transpose(1, 2), z).squeeze()
        term_2 = -self.cov_log_det_
        logprob = term_1 + term_2
        
        return logprob

class ConstructorA(nn.Module):
    # construct module a from a_kwargs
    def __init__(self, loc_classname, loc_kwargs):
        nn.Module.__init__(self)
        self.loc = eval(loc_classname)(**loc_kwargs)
        self.scale_vec = None
        
    def forward(self, *args):
        lc = self.loc(*args)
        return lc

class ConstructorB(nn.Module):
    # construct module b from b_kwargs
    def __init__(self, loc_classname, loc_kwargs):
        nn.Module.__init__(self)
        self.loc = eval(loc_classname)(**loc_kwargs)
        self.scale_vec = None
        
    def forward(self, *args):
        lc = self.loc(*args)
        return lc

class ConstructorC(nn.Module):
    # Construct module c from c_kwargs
    def __init__(self, loc_classname, loc_kwargs, gauss_dim):
        nn.Module.__init__(self)
        self.loc = eval(loc_classname)(**loc_kwargs)
        self.gauss_dim = gauss_dim
        vec_dim = gauss_dim + gauss_dim*(gauss_dim+1)//2
        self.inds = self.vec_to_inds(gauss_dim, vec_dim)

    def forward(self, *args):
        lc = self.loc(*args)
        return Gaussian(self.gauss_dim, lc, self.inds)        

    def vec_to_inds(self, x_dim, vec_dim):
        """Computes the indices of scale_tril coeffs,
        scale_tril is filled main diagonal first

        x_dim: dimension of the random variable
        vec_dim: dimension of the vector containing
                 the coeffs of loc and scale_tril
        """
        ldiag, d, c = x_dim, 0, 0  # diag length, diag index, column index
        inds = [[], []]  # list of line and column indexes
        for i in range(vec_dim - x_dim):  # loop over the non-mean coeff
            inds[0].append(c+d)  # line index
            inds[1].append(c)  # column index
            if c == ldiag-1:  # the current diag end is reached
                ldiag += -1  # the diag length is decremented
                c = 0  # the column index is reinitialized
                d += 1  # the diag index is incremented
            else:  # otherwize, only the column index is incremented
                c += 1
        return inds    

class ConstructorProp(nn.Module):
    # construct propogator based on prop_kwargs
    # return a Gaussian made from a vector, 
    # this vector is made of the concatnation of loc and scale_vec
    def __init__(self, loc_classname, loc_kwargs,
                 gauss_dim, scale_vec_classname, scale_vec_kwargs):
        nn.Module.__init__(self)       
        self.gauss_dim = gauss_dim # same as x_dim
        self.loc = eval(loc_classname)(**loc_kwargs) # init e.g. Lin2d module
        self.scale_vec = eval(scale_vec_classname)(**scale_vec_kwargs) # Cst module

    def forward(self, *args):
        lc = self.loc(*args)
        sc = self.scale_vec(*args)
        return GaussianDiag(self.gauss_dim, torch.cat((lc, sc), dim=1))        

class ConstructorObs(nn.Module):
    # construct observor based on obs_kwargs
    # return a Gaussian made from a vector, 
    # this vector is made of the concatnation of loc and scale_vec
    def __init__(self, loc_classname, loc_kwargs,
                 gauss_dim=None,
                 scale_vec_classname=None, scale_vec_kwargs=None):
        nn.Module.__init__(self)
        self.gauss_dim = gauss_dim
        self.loc = eval(loc_classname)(**loc_kwargs) # init e.g. Id module
        self.scale_vec = eval(scale_vec_classname)(**scale_vec_kwargs) # Cst module       
    def forward(self, *args):
        lc = self.loc(*args)
        sc = self.scale_vec(*args)
        return GaussianDiag(self.gauss_dim, torch.cat((lc, sc), dim=1))    
    
