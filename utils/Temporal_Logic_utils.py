# import the necessary packages
#from matplotlib.font_manager import _Weight
#from types import NoneType
from select import select
from unittest import result
#from matplotlib.font_manager import _Weight
import numpy as np
from operator import xor
import torch
# Malisiewicz et al.
MAX_LIMIT =1000.000


def Segment(scores):

    eta = 3.0
    B, C, W = scores.size()
    value, indix = torch.max(scores,2)
    value = value.view(B,C,1)    
    result_metrics = torch.tensor([])
    for delta in range(2,9):

        boolean_scores = torch.gt(scores,delta*value/10.0)
        false_extend  = torch.zeros(B,C,1,dtype=torch.bool)

        shift_right = torch.cat((boolean_scores,false_extend),2)
        shift_left  = torch.cat((false_extend,boolean_scores),2)
        idx_bool = torch.logical_xor(shift_left,shift_right)
        idx_boole = torch.logical_and(idx_bool,shift_left)
        idx_bools = torch.logical_and(idx_bool,shift_right)

        nonzero_start = torch.nonzero(idx_bools)
        nonzero_end   = torch.nonzero(idx_boole)

        duration = nonzero_end-nonzero_start

        new_start = torch.cat((nonzero_start,duration),1)
        
        for i in range(nonzero_end.size(0)):
            select_int = scores[nonzero_start[i,0],nonzero_start[i,1],nonzero_start[i,2]:nonzero_end[i,2]]
            
            var_mean = torch.var_mean(select_int,unbiased=False)
            sum_duration =  torch.sum(select_int)
            new_start[i,3] = 0.1*torch.sqrt(delta)*sum_duration+var_mean[0] + eta*var_mean[1]   # score value
            new_start[i,4] = nonzero_start[i,2]+(var_mean[0]/var_mean[1]).int()    #adjust start
            new_start[i,5] = duration[i,2]+(var_mean[1]/var_mean[0]).int()             # adjust duration

        result_metrics = torch.cat((result_metrics,new_start),0)




    select_one = torch.zeros(scores.size())
    select_two = torch.zeros(scores.size())
    for i in range(B):
        for j in range(C):
            start_ij = result_metrics[(result_metrics[:,0]==i) & (result_metrics[:,1]==j),:]
            metric = start_ij[:,3]
            max_value, sort_idx  = torch.sort(metric,descending=True)
            first, second = sort_idx[0], sort_idx[1]
            select_one[i,j,start_ij[first,4]:start_ij[first,4]+start_ij[first,5]] = scores[i,j,start_ij[first,4]:start_ij[first,4]+start_ij[first,5]]
            select_two[i,j,start_ij[second,4]:start_ij[second,4]+start_ij[second,5]] = scores[i,j,start_ij[second,4]:start_ij[second,4]+start_ij[second,5]]
            


    return [select_one, select_two]


 

def basin_gen(scores):

    result =[]
    for i in range(2,9):
        select = Segment(scores, i/10.0)
        result.append(select)


    return result




def gen_proposal(scores):

    eta = 3.0


    B, C, W = scores.size()
    value, indix = torch.max(scores,2)
    value = value.view(B,C,1)
    boolean_scores = torch.gt(scores,0.7*value)


    false_extend  = torch.zeros(B,C,1,dtype=torch.bool)

    shift_right = torch.cat((boolean_scores,false_extend),2)
    shift_left  = torch.cat((false_extend,boolean_scores),2)
    idx_bool = torch.logical_xor(shift_left,shift_right)
    idx_boole = torch.logical_and(idx_bool,shift_left)
    idx_bools = torch.logical_and(idx_bool,shift_right)

    nonzero_start = torch.nonzero(idx_bools)
    nonzero_end   = torch.nonzero(idx_boole)

    duration = nonzero_end-nonzero_start

    new_start = torch.cat((nonzero_start,duration),1)
    
    for i in range(nonzero_end.size(0)):
        select_int = scores[nonzero_start[i,0],nonzero_start[i,1],nonzero_start[i,2]:nonzero_end[i,2]]
        
        var_mean = torch.var_mean(select_int,unbiased=False)
        sum_duration =  torch.sum(select_int)
        new_start[i,3] = sum_duration- eta*var_mean[0] + var_mean[1]
        new_start[i,4] = nonzero_start[i,2]+(var_mean[0]/2.0).int()
        new_start[i,5] = duration[i,2]-var_mean[0].int()


    select = torch.zeros(scores.size())
    for i in range(B):
        for j in range(C):
            start_ij = new_start[(new_start[:,0]==i) & (new_start[:,1]==j),:]
            metric = start_ij[:,3]
            max_idx  = torch.argmax(metric)
            select[i,j,start_ij[max_idx,4]:start_ij[max_idx,4]+start_ij[max_idx,5]] = scores[i,j,start_ij[max_idx,4]:start_ij[max_idx,4]+start_ij[max_idx,5]]
            


    return select



def Always(wrho):
    #weight = torch.sigmoid(weight)
    #wrho     = torch.mul(weight,robustness)
    wrho     =wrho.double()
    pos_wrho = torch.where(wrho>=0.0,1+wrho,1.0)
    pos_prod = torch.prod(pos_wrho,-1)
    pos_prod =pos_prod.double()
    #print(pos_prod)
    pos_prod =torch.where(pos_prod<MAX_LIMIT,pos_prod,MAX_LIMIT*torch.sigmoid(pos_prod) )        
    #pos_prod[pos_prod== float("Inf")]=MAX_LIMIT
    #pos_prod[pos_prod== float("nan")]=0.0
    #print(pos_prod)
    pos_num  = torch.where(wrho>0,1.0,0.0)
    power    = torch.sum(pos_num,-1)+0.01
    #print(power)
    
    
    pos_result = pos_prod**(1/power)-1
    pos_result =pos_result.double()
    neg_whro = torch.where(wrho<0.0,-wrho,0.0)
    neg_sum  = torch.sum(-neg_whro,-1)
    neg_result = neg_sum/power
    result = torch.where(neg_result<0,neg_result,pos_result)



    return result


def Eventually(wrho):

    #wrho     = torch.mul(weight,robustness)
    wrho     = wrho.double()
    neg_wrho = torch.where(wrho<=0.0,1-wrho,1.0)
    neg_prod = torch.prod(neg_wrho,-1)
    neg_prod = neg_prod.double()
    neg_num  = torch.where(wrho<=0,1.0,0.0)
    neg_prod = torch.where(neg_prod<MAX_LIMIT,neg_prod,MAX_LIMIT*torch.sigmoid(neg_prod))
    
    power    = torch.sum(neg_num,-1)+0.01

    neg_result = -neg_prod**(1/power)+1

    pos_wrho = torch.where(wrho>0,wrho,0.0)
    pos_sum  = torch.sum(pos_wrho,-1)
    pos_result = pos_sum/power

    result = torch.where(pos_result>0,pos_result,neg_result)
    #print('eventually output',result)
    #print('grad eventually result', result.grad)
    return result



