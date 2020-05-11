import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

import math
import random
from torchtext import data, datasets, vocab

import os

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, target, preds_down,preds_middle,preds_up):
        preds = torch.cat([preds_down,preds_middle,preds_up], axis=2)
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target[:,:,0] - preds[:,:,i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

def loss_fct(mu, sigma, labels, loss_kind):
    if loss_kind=='nll_dist_normal':
        indices= (labels != 0)
        log_likelihood = torch.distributions.Normal(mu[indices], sigma[indices]).log_prob(labels[indices])
        loss = -log_likelihood.mean()
    elif loss_kind  == 'nll_dist_lognormal':
        indices= (labels != 0)
        log_likelihood = torch.distributions.log_normal.LogNormal(mu[indices], sigma[indices]).log_prob(labels[indices])
        loss = -log_likelihood.mean()
    elif loss_kind == 'quantileloss_dist_normal':
        lossQ = QuantileLoss([0.1587,0.5,0.8413])
        preds_middle = mu
        preds_downs =  preds_middle - sigma
        preds_up = preds_middle + sigma
        loss = lossQ(labels[:,:,None],preds_downs,preds_middle,preds_up)
    return loss


def evaluate(model, x_test, tokens_tst_idx, v_tst, windows_size, sampling = False, number_of_samples = 100, scaled_param = False):
    """
    Function to apply the model on a batch and get the evaluation
    model : The model to use for the evaluation
    x_test: 3D array, 
    tokens_tst_idx, v_tst, windows_size, sampling = False, number_of_samples = 100, scaled_param = False
    
    """
    batch_size = x_test.shape[0]
    # v_tst is expanded, which means it has the shape (batch_size, windows_size, 2)
    
    if sampling :
        # Sample from the distribution to estimate
        samples = torch.empty((number_of_samples,batch_size, windows_size,1))
        
        # Distribution parameters
        mu, sigma = model(x = x_test, tokens = tokens_tst_idx)
        
        scaled_mu = mu[:,:,0] * v_tst[:,:,0] + v_tst[:,:,1]
        scaled_mu = scaled_mu[:,:,None]

        scaled_sigma = sigma[:,:,0] * v_tst[:,:,0]
        scaled_sigma = scaled_sigma[:,:,None]
        
        # Sampling loop
        distributions = torch.distributions.log_normal.LogNormal(mu, sigma)
        for k in range(number_of_samples):
            sample = distributions.sample()
            if scaled_param :
                sample = sample[:,:,0]* v_tst[:,:,0] + v_tst[:,:,1]
            sample = sample[:,:,None]
            samples[k,:,:] = sample
        
        estimation = samples.median(dim=0)[0] * v_tst[:,:,:1] + v_tst[:,:,1:]
        
        if scaled_param :
            return samples, estimation, scaled_mu,scaled_sigma
        else :
            return samples, estimation, mu, sigma
    
    else:
        # Distribution parameters
        mu, sigma = model(x = x_test, tokens = tokens_tst_idx)
        
        scaled_mu = mu[:,:,0] * v_tst[:,:,0] + v_tst[:,:,1]
        scaled_mu = scaled_mu[:,:,None]

        scaled_sigma = sigma[:,:,0] * v_tst[:,:,0]
        scaled_sigma = scaled_sigma[:,:,None]
        
        if scaled_param :
            estimation = scaled_mu
        else:
            estimation = mu
        return estimation, scaled_mu, scaled_sigma


def test_evaluation(model,X_tst_check, tokens_tst_idx, v_tst, labels_tst, windows_size, first_affect=False, sampling=True, number_of_samples=25,loss_kind = 'nll_dist_normal'):
    """
    Function to compute the predictions and the losses for the test batchs
    # X_tst_check :  (batch, global_window, vars) : the batch to manage
    # tokens_tst_idx : (batch,) : The list of tokens of the batchs
    # v_tst :  (batch,2) :  The scale of serie for each batch
    # labels_tst : The serie to predict
    # windows_size: The model window size
    # first_affect : {True, False} : To make the next prediction based on the first window predicted values or not (using the authentic ones)
    # sampling : {True, False} : To make predictions based on samling or not (the mean of the distribution is used)
    # number_of_samples : If using sampling, the number of samples
    
    returns : 
    # predicted_serie :  (batch, global_window, 1) : The predicted serie
    # v_tst :  (batch,2) :  The scale of serie for each batch
    # labels_tst : (batch, global_window) : The true value of the serie
    # tst_loss : the mean loss
    # mus : (windows_total,batch,window_size,1) : The predicted unscaled mean predicted by the model
    # sigmas : (windows_total,batch,window_size,1) : The predicted unscaled sigma predicted by the model
    """

    # Parameters :
    tst_global_window_size = X_tst_check.shape[1] # which is the train_windows_size that come in the batch == 62
    batch_size = X_tst_check.shape[0]
    stride = 1 # it's 1 to predict next step, if not (e.g 4), there would be stride-1 (e.g 4-1) instant not predicted each time
    windows_total = (tst_global_window_size - (windows_size-stride))//stride # total number of windows

    # Evaluation :
    sampling = sampling
    number_of_samples = number_of_samples

    st = 0
    nd = st + windows_size

    mus = torch.empty((int(windows_total),batch_size, windows_size,1))
    sigmas= torch.empty((int(windows_total),batch_size, windows_size,1))
    
    predicted_serie = torch.empty((batch_size, tst_global_window_size,1))


    for i in range(int(windows_total)):

        # Xtest extraction
        x_test = X_tst_check[:,st:nd,:]

        # Model application :

        ### Expand the v_tst to the shape (batch, windows_size,2), since the scale is the same  for all instants in the window (due to the small size difference between the learning window and the test window)
        v_tst_expanded = v_tst.expand(windows_size,batch_size,2).transpose(1,0)
        if sampling:
            samples, estimation, mu, sigma = evaluate(model, x_test, tokens_tst_idx, v_tst_expanded, windows_size, 
                                     sampling, number_of_samples)
        else:
            estimation, mu, sigma = evaluate(model, x_test, tokens_tst_idx, v_tst_expanded, windows_size, 
                                     sampling, number_of_samples)
        # mu & sigma history
        mus[i,:,:,:] = mu
        sigmas[i,:,:,:] = sigma

        # Changing the value in the X_tst to the predicted next value to make next prediction:
        
        if i == 0 and first_affect: # the first window, we affect all the predicted elements, so next prediction would be based on those
            X_tst_check[:,st:nd,0] = estimation[:,st:nd,0]
            
        if nd < tst_global_window_size :
            X_tst_check[:,nd,0] = estimation[:,-1,0]
        
        # Predicted elements :
        
        if i == 0 :
            predicted_serie[:,st:nd,0] = estimation[:,st:nd,0] # predicted affected
        
        if nd < tst_global_window_size :
            predicted_serie[:,nd,0] = estimation[:,-1,0]

        # Loss calculation :
        tst_loss = loss_fct(mu, sigma, labels_tst[:,st:nd],loss_kind)
        st+=stride
        nd+=stride
        
    tst_loss = tst_loss/int(windows_total)
    return predicted_serie, v_tst, labels_tst, tst_loss, mus, sigmas