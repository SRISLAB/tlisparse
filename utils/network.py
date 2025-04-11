#
# $Id: modular.py 740 2021-07-20 13:54:15Z hat $
#
from statistics import mean
import math
import random
import numpy as np
LENGTH_SIGNAL = 266
WIGHT_THR  =0.05

def relu(x):
    if x>0.1:
        return x
    else:
        return 4*np.exp(x-0.1)-0.8

    return max(0.1,x)

def diff_relu(x):
    if x>0.1:
        return 1
    else:
        return 0.8*np.exp(x-0.1)

def signfunction(x,ispositive=True):
    if ispositive:
        return max(0.0,x)
    else:
        return -signfunction(-x,True)
        
vsignfunction = np.vectorize(signfunction)

def signfunction_w(x,rho, ispositive=True):
    if ispositive:
        result = max(0.0, x*rho)/(rho+0.000000000001)
        return result 
    else:
        return -signfunction_w(x, -rho, True)

vsignfunction_w = np.vectorize(signfunction_w)


def quant(x, length):
    #print 'output of encoder', x 
    result = round(x*length)
    if result>length:
        return length
    else:
        return result 
vquant = np.vectorize(quant)
def tanh(x):
    if abs(x)>10:
        x = np.sign(x)*10.0

 
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

def diff_quant(x, length, kappa):

    tan = 1-tanh(x-(round(x)+0.5)/length)**2

    return tan/(2*length*tanh(0.5*kappa/length))

vdiff_quant = np.vectorize(diff_quant)

def sigmoid(x):
    if abs(x)>10:
        x = np.sign(x)*10.0
 
    return 1/(1 + math.exp(-x))

vsigmoid = np.vectorize(sigmoid)
def sigmoid_shift(x):
    if abs(x)>10:
        x = np.sign(x)*10.0
    return 1/(1 + math.exp(-x))-0.5
vshift_sigmoid = np.vectorize(sigmoid_shift)
def diff_sigmoid(x):
    if abs(x)>10:
        x = np.sign(x)*10.0
    return sigmoid(x)*(1-sigmoid(x))

vdiff_sigmoid = np.vectorize(diff_sigmoid)

class Always():
    def __init__(self, weight,  tau1,tau2, bias=np.array([]), islarge=True):
        self.tau1 = tau1
        self.tau2 = tau2
        self.bias = bias
        self.islarge = islarge
        self.type ='always'
        self.w =weight 
        self.rho = None 
        self.activation = 'Linear'
        
    def set_weight(self, weight):
        self.w  = weight

    def set_activation(self,act):
        self.activation= act 
 
    def set_rho(self,inputs):
        if self.islarge:
            rho = inputs -self.bias
        else:
            rho = self.bias - inputs 

        self.rho = rho 
  
    def set_bias(self, bias):
        self.bias = bias 

    def robustness(self, weight, inputs):
        self.set_rho(inputs)
        rho = self.rho
        wrho = np.multiply(weight, rho)
        if self.activation =='ReL':
            zero = np.arrary([0]*len(rho))
            wrho = np.fmax(wrho,zero)

        ewrho = wrho[self.tau1:self.tau2+1]
        if min(ewrho)>=0:
            result = wrho[self.tau1:self.tau2+1]+1
            result = np.product(result)
            result = result**(1.0/(self.tau2-self.tau1+1)) -1.0
        else:
            sigrho = vsignfunction(ewrho, False)
            result = sum(sigrho)
            result = result/(self.tau2-self.tau1+1)

        return result


    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    def gradient_w(self):
        wrho = np.multiply(self.w, self.rho)


        if self.activation =='ReL':
            zero = np.arrary([0]*len(rho))
            wrho = np.fmax(wrho,zero)

        M = self.tau2-self.tau1+1
        ewrho = wrho[self.tau1:self.tau2+1]
        if min(ewrho)>=0:
            shift = ewrho+1
            result = np.product(shift)
            temp = result/shift 
            result = result**(1.0/M-1) 

            result = result*np.multiply(self.rho[self.tau1:self.tau2+1],temp)/M
        else:
            result = vsignfunction_w(self.rho[self.tau1:self.tau2+1],self.w[self.tau1:self.tau2+1], False)/M

        

        ones = np.array([1]*len(ewrho))
        idx = np.fmin(ewrho*10000000, ones)
        result = np.multiply(idx,result)
        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 


    def gradient_r(self):

        wrho = np.multiply(self.w, self.rho)
        if self.activation =='ReL':
            zero = np.arrary([0]*len(rho))
            wrho = np.fmax(wrho,zero)
        ewrho=wrho[self.tau1:self.tau2+1]

        result = 1.0
        M = self.tau2-self.tau1+1
        if min(ewrho)>=0:
            shift = ewrho +1
            result = np.product(shift)
            temp = result/shift
            result = result**(1.0/M-1) 
            result = result*np.multiply(self.w[self.tau1:self.tau2+1], temp)/M 
        else:
            result = vsignfunction_w(self.w[self.tau1:self.tau2+1], self.rho[self.tau1:self.tau2+1], False)/M


        ones = np.array([1]*len(ewrho))
        idx = np.fmin(ewrho*10000000, ones)

        result = np.multiply(idx,result)



        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 


    def gradient_p(self):
        if self.islarge:
            return -mean(self.gradient_r())
        else:
            return mean(self.gradient_r())

class Eventually():
    def __init__(self, weight,  tau1,tau2, bias=np.array([]), islarge=True):
        self.tau1 = tau1
        self.tau2 = tau2
        self.type = 'eventually'        
        self.w =weight 
        self.bias = bias 
        self.islarge = islarge
        self.rho = None 
        self.activation = 'Linear'      

    def set_weight(self, weight):
        self.w = weight


    def set_bias(self, bias):
        self.bias = bias 


    def set_activation(self,act):
        self.activation= act 

    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 

    def set_rho(self,inputs):
        if self.islarge:
            rho = inputs -self.bias
        else:
            rho = self.bias - inputs 
        self.rho = rho 

    def robustness(self, weight, inputs):
        self.set_rho(inputs)
        rho = self.rho
        wrho = np.multiply(weight,rho)
        if self.activation =='ReL':
            zero = np.arrary([0]*len(rho))
            wrho = np.fmax(wrho,zero)

        ewrho = wrho[self.tau1:self.tau2+1]
        #print wrho, weight, rho 
         
        if max(ewrho)<0:
 
            result = 1-ewrho
            result = np.product(result)
            result = -result**(1.0/(self.tau2-self.tau1+1)) +1.0

        else:
            sigrho = vsignfunction(ewrho, True)
            result = sum(sigrho)
            result = result/(self.tau2-self.tau1+1)

        return result


    def gradient_w(self):

        wrho = np.multiply(self.w, self.rho)
        if self.activation =='ReL':
            zero = np.arrary([0]*len(rho))
            wrho = np.fmax(wrho,zero)

        ewrho = wrho[self.tau1:self.tau2+1]

        M = self.tau2-self.tau1+1

        if max(ewrho)<0:
            shift = 1 - ewrho
            result = np.product(shift)
            temp = result/shift
            result = result**(1.0/M-1)
            result = result*np.multiply(self.rho[self.tau1:self.tau2+1],temp)/M

        else:
            result =  vsignfunction_w(self.rho[self.tau1:self.tau2+1], self.w[self.tau1:self.tau2+1],  True)/M



        ones = np.array([1]*len(ewrho))
        idx = np.fmin(ewrho*10000000, ones)

        result = np.multiply(idx,result)



        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 


    def gradient_r(self):


        wrho = np.multiply(self.w, self.rho)
        if self.activation =='ReL':
            zero = np.arrary([0]*len(rho))
            wrho = np.fmax(wrho,zero)

        ewrho = wrho[self.tau1:self.tau2+1]

        result = 1.0
        M = self.tau2-self.tau1+1

        if max(ewrho)<0:
 
            shift  = 1- ewrho
            result = np.product(shift)
            temp = result/shift
            result = result**(1.0/M-1)
            result = result*np.multiply(self.w[self.tau1:self.tau2+1], temp)/M

        else:
            result = vsignfunction_w(self.w[self.tau1:self.tau2+1],self.rho[self.tau1:self.tau2+1], True)/M

        ones = np.array([1]*len(ewrho))
        idx = np.fmin(ewrho*10000000, ones)

        result = np.multiply(idx,result)

        grad = np.zeros(self.w.size)
        grad[self.tau1:self.tau2+1] = result

        return grad 

    def gradient_p(self):
        if self.islarge:
            return -mean(self.gradient_r())
        else:
            return mean(self.gradient_r())



class EventualAlways():

    def __init__(self, weight, tau0, tau1,tau2, bias=np.array([]), islarge=True):
        self.tau0 = tau0
        self.tau1 = tau1
        self.tau2 = tau2

        self.type = 'eventualalways'
        self.islarge = islarge

        self.w =weight 
        self.bias = bias 
        self.rho = None 



    def set_bias(self, bias):
        self.bias = bias 
    def robustness(self, weight, inputs):
        self.set_rho(inputs)

        rho = self.rho
        erho =[]
        for i in range(0,self.tau0+1):
            new_tau1 = min(self.tau1+i,len(weight)-1)
            new_tau2 = min(self.tau2+i, len(weight)-1)
            alw = Always(weight, new_tau1, new_tau2,0.0, True)
            alw.set_rho(rho)
            rob = alw.robustness(weight,rho)
            erho.append(rob)

        weights = np.array([1]*len(erho))
        Or = OR(weights, np.array(erho))


        return Or.robustness(weights,Or.rho)

    def set_interval(self, tau1, tau2):
        self.tau1 = tau1
        self.tau2 = tau2

    def set_weight(self, weight):
        self.w =weight


    def set_rho(self,inputs):
        if self.islarge:
            rho = inputs -self.bias
        else:
            rho = self.bias - inputs 

        self.rho = rho 

    def set_shift(self,tau0):
        self.tau0 = tau0
 

    def gradient_w(self):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)           
            alw = Always(self.w,new_tau1, new_tau2,0.0, True)
            alw.set_rho(self.rho)
            rhos.append(alw.robustness(self.w,self.rho))

        weights = np.array([1]*len(rhos))

        Or = OR(weights, np.array(rhos))
        grad_or =  Or.gradient_r()
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            alw = Always(self.w, new_tau1, new_tau2,0.0,True)
            alw.set_rho(self.rho)
            result = result +grad_or[k]*alw.gradient_w()

        return result

    def gradient_r(self):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            alw = Always(self.w, new_tau1,new_tau2,0.0, True)
            alw.set_rho(self.rho)
            rhos.append(alw.robustness(self.w,self.rho))

        weights = np.array([1]*len(rhos))
        Or = OR(weights,np.array(rhos))
        grad_or =Or.gradient_r()

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)    
            alw = Always(self.w, new_tau1,new_tau2,0.0, True)
            alw.set_rho(self.rho)
            result = result + grad_or[k]*alw.gradient_r()

        return result

    def gradient_p(self):
        if self.islarge:
            return -mean(self.gradient_r())
        else:
            return mean(self.gradient_r())


class AlwaysEventual():
    def __init__(self, weight, tau0, tau1,tau2, bias=np.array([]), islarge=True):
        self.tau0 = tau0
        self.tau1 = tau1
        self.tau2 = tau2
        self.type ='alwayseventual'
        self.w =weight 
        self.bias = bias 
        self.islarge = islarge
        self.rho = None 
   

    def set_weight(self, weight):
        self.w = weight

    def set_bias(self, bias):
        self.bias = bias 

    def robustness(self, weight, inputs):
        self.set_rho(inputs)
        rho = self.rho

        erho =[]
        for i in range(0,self.tau0+1):
            new_tau1 = min(self.tau1+i,self.w.size-1)
            new_tau2 = min(self.tau2+i, self.w.size-1)
            even = Eventually(weight, new_tau1,new_tau2,0.0, True)
            even.set_rho(self.rho)
            rob = even.robustness(weight,self.rho)
            erho.append(rob)

        weights =np.array([1]*len(erho))

        And = AND(weights, np.array(erho))

        return And.robustness(weights,And.rho)



    def set_interval(self,tau1,tau2):
        self.tau1 = tau1 
        self.tau2 = tau2 


    def set_rho(self,inputs):
        if self.islarge:
            rho = inputs -self.bias
        else:
            rho = self.bias - inputs 
        self.rho = rho 
    
    def set_shift(self,tau0):
        self.tau0 = tau0

    def gradient_w(self):
        result =0.0
        rhos =[]
 
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)    
            even = Eventually(self.w,new_tau1,new_tau2,0.0, True)
            rhos.append(even.robustness(self.w,self.rho))

        weights = np.array([1.0]*len(rhos))

        And = AND(weights, np.array(rhos))
        grad_and = And.gradient_r()

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            even = Eventually(self.w,new_tau1,new_tau2,0.0, True)
            even.set_rho(self.rho)
            result = result + grad_and[k]*even.gradient_w()

        return result
    def gradient_r(self):
        result =0.0
        rhos =[]
        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)    
            even = Eventually(self.w,new_tau1,new_tau2,0.0, True)
            
            rhos.append(even.robustness(self.w,self.rho))
        

        weights = np.array([1.0]*len(rhos))
        And = AND(weights,np.array(rhos))
        grad_and = And.gradient_r()

        for k in range(self.tau0+1):
            new_tau1 = min(self.tau1+k,self.w.size-1)
            new_tau2 = min(self.tau2+k, self.w.size-1)                
            even = Eventually(self.w,new_tau1,new_tau2,0.0, True)
            even.set_rho(self.rho)
            result = result +grad_and[k]*even.gradient_r()
        return result


    def gradient_p(self):
        if self.islarge:
            return -mean(self.gradient_r())
        else:
            return mean(self.gradient_r())

class AND():
    def __init__(self, w, rho=np.array([])):
        self.rho = rho
        self.w = w
        self.type = 'and'
        self.activation = 'Linear' 


    def set_weight(self, weight):
        self.w = weight
    def set_activation(self, activation):
        self.activation = activation 
    def set_rho(self,rho):
        self.rho = rho 

    def robustness(self, weight, inputs):
        self.set_rho(inputs)
        weig = vshift_sigmoid(weight)
        alw = Always(weig, 0,weight.size-1,0.0,True)
        alw.set_activation(self.activation)

        return alw.robustness(weig,self.rho)

    def gradient_w(self):

        alw = Always( vshift_sigmoid(self.w), 0, self.w.size-1,0.0, True)
        alw.set_rho(self.rho)
  

        alw.set_activation(self.activation)
        

        result =np.multiply(alw.gradient_w(), vdiff_sigmoid(self.w))


        return result 

    def gradient_r(self):
        
        alw = Always( vshift_sigmoid(self.w), 0, self.w.size-1,0.0, True)
        alw.set_activation(self.activation)
        alw.set_rho(self.rho)

        result = alw.gradient_r()

            #print 'not satisfied', result
        return result

class OR():

    def __init__(self,  w, rho=np.array([])):
        self.rho = rho
        self.w = w
        self.type = 'or'
        self.activation = 'Linear' 

    
    def robustness(self, weight, inputs):
        self.set_rho(inputs)

        rho = self.rho
        weig = vshift_sigmoid(weight)
        evn = Eventually(weig, 0, weight.size-1, 0.0, True)
        evn.set_activation(self.activation) 
  

        return evn.robustness(weig,rho)

    def set_type(self,activation):
        self.activation = activation
    def set_weight(self, weight):
        self.w = weight


    def set_rho(self,rho):

        self.rho = rho 

    def gradient_w(self):

        evn = Eventually( vshift_sigmoid(self.w), 0, self.w.size-1,0.0, True)
        evn.set_activation(self.activation)
        evn.set_rho(self.rho)
        result = np.multiply(evn.gradient_w(), vdiff_sigmoid(self.w))
        return result 

    def gradient_r(self):


        evn = Eventually(vshift_sigmoid(self.w), 0, self.w.size-1, 0.0, True)
        evn.set_activation(self.activation)
        evn.set_rho(self.rho)
        result = evn.gradient_r()

        return result   


class Ddecoder():
    def __init__(self, W_o, W_h):

        self.W_o = W_o 
        self.W_h = W_h 
        self.h_out  = None 

    def set_weight(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h         

    def output(self, inputs):
        weight =[]
        h_out =[]
        for w in self.W_h:
            z = np.multiply(w,inputs)

            h_out.append(relu(mean(z)))
        self.h_out = np.array(h_out)
        for w in self.W_o:
            z = np.multiply(self.h_out,w)
            weight.append(relu(mean(z)))
        return  np.array(weight)

    def gradient_o(self, i,  h_out):
        # i the ith neuron in output layer
     
        z = np.multiply(self.h_out,self.W_o[i])
        result= h_out*diff_relu(mean(z))/h_out.size
  
        return result
    def gradient_h(self, j,  inputs, h_out, Grad_o):
        # jth neuron in hidden

        result = 0.0
        w = np.array(self.W_o)[:,j]
        if abs(h_out[j])>0:

            grad_o = np.array(Grad_o)[:,j]/h_out[j]
        else:
            grad_o = np.array(Grad_o)[:,j]

        result =np.multiply(w,grad_o)
        zh = np.multiply(inputs,self.W_h[j]) 

        #print 'diff', diff_sigmoid(mean(zh)), 'mean', mean(zh), 'result', sum(result)
        result = sum(result)*diff_relu(mean(zh))*inputs

        return result

    def gradient_i(self,i, inputs,  Grad_h):
        #ith neuron in input
        w = np.array(self.W_h)[:,i]
        grah_h = np.array(Grad_h)[:,i]
        if abs(inputs[i])>0:
            result = np.multiply(w,grah_h)/inputs[i]
        else:
            result = np.multiply(w,grah_h)


        return sum(result)

class Dencoder():
    def __init__(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h
 
        self.h_out =None 
        self.out = None 

    def set_weight(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h        


    def output(self, inputs):
        h_out =[]
        for w in self.W_h:
 
            result = np.multiply(inputs,w)
            h_out.append(relu(mean(result)))
        self.h_out = np.array(h_out)
        out = []
        for w in self.W_o:
            result = np.multiply(w,self.h_out)
            out.append(relu(mean(result)))
        self.out = np.array(out) 

        return np.array(out) 


    def gradient_o(self, i,  h_out):
        # ith output neuron
        z = np.multiply(self.h_out,self.W_o[i])
        result= h_out*diff_relu(mean(z))/h_out.size
        return result

    def gradient_h(self, j,  inputs,h_out, Grad_o):
        result = 0.0
        w = np.array(self.W_o)[:,j]
        if abs(h_out[j])> 0:
            grad_o = Grad_o[:,j]/h_out[j]
        else:
            grad_o = Grad_o[:,j]

        result =np.multiply(w,grad_o)
        zh = np.multiply(inputs,self.W_h[j]) 
        result = sum(result)*diff_relu(mean(zh))*inputs/zh.size

        return result


class Decoder():
    def __init__(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h
        self.h_out =None 
    def set_weight(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h        
    def output(self, inputs):
        weight =[]
        h_out =[]
        for w in self.W_h:
            z = mean(np.multiply(w,inputs))/LENGTH_SIGNAL
            h_out.append(sigmoid(z))

        self.h_out = np.array(h_out)

        for w in self.W_o:

            z = np.multiply(self.h_out,w)

            weight.append(sigmoid(mean(z)))

        return  np.array(weight)


    def gradient_o(self, i,  h_out):
        # i the ith neuron in output layer

        z = np.multiply(self.h_out,self.W_o[i])
        result= h_out*diff_sigmoid(mean(z))/h_out.size
        return result

    def gradient_h(self, j,  tau, h_out, Grad_o):
        # jth neuron in hidden

        result = 0.0
        w = np.array(self.W_o)[:,j]
        if abs(h_out[j]) >0.0:
            grad_o = Grad_o[:,j]/h_out[j]
        else:
            grad_o = Grad_o[:,j]

        result =np.multiply(w,grad_o)
        zh = np.multiply(tau,self.W_h[j]) 
   
        result = mean(result)*diff_sigmoid(mean(zh))*tau 


        return result

    def gradient_i(self,i, tau,  Grad_h):
        #ith neuron in input
        w = np.array(self.W_h)[:,i]
        grah_h = np.array(Grad_h)[:,i]
        if abs(tau[i]) > 0.0:
            result = np.multiply(w,grah_h)/tau[i]
        else:
            result = np.multiply(w,grah_h)


        return mean(result)
        
class Encoder():
    def __init__(self, W_o, W_h, qunti_length, kappa):
        self.W_o = W_o
        self.W_h = W_h
        self.qunti_length = qunti_length
        self.kappa = kappa
        self.h_out =None 
        self.out = None 


    def set_weight(self, W_o, W_h):
        self.W_o = W_o
        self.W_h = W_h        


    def output(self, rho):
        h_out =[]
        for w in self.W_h:
            result = np.multiply(rho,w)
            h_out.append(tanh(mean(result)))
        self.h_out = np.array(h_out)
        out = []
        for w in self.W_o:
            result = np.multiply(w,self.h_out)
            out.append(sigmoid(mean(result)))
        self.out = np.array(out) 

        tau1 = min(quant(out[0],self.qunti_length),self.qunti_length-2)
        tau2 = min(tau1+quant(out[1],self.qunti_length),self.qunti_length-1)

 

        assert tau2>=tau1 
        return np.array([tau1,tau2,out[2]])


    def gradient_o(self, i,  h_out):
        # ith output neuron
        if i ==0:
            diff_q1 = diff_quant(self.out[0],self.qunti_length,self.kappa)
            diff_q2 = diff_quant(self.out[1],self.qunti_length,self.kappa)
            z = np.multiply(self.h_out,self.W_o[i])
            result= h_out*diff_sigmoid(mean(z))*(diff_q1+diff_q2)/LENGTH_SIGNAL

        elif i==1:
            diff_q2 = diff_quant(self.out[1],self.qunti_length,self.kappa)
            z = np.multiply(self.h_out,self.W_o[i])
            result= h_out*diff_sigmoid(mean(z))*diff_q2/LENGTH_SIGNAL

        else:
            z = np.multiply(self.h_out,self.W_o[i])
            result= h_out*diff_sigmoid(mean(z))

       
        
        return result 


    def gradient_h(self, j,  inputs,h_out, Grad_o):
        w = np.array(self.W_o)[:,j]
        grad_o = np.array(Grad_o[:,j])/h_out[j]
        
        result =np.multiply(w,grad_o)/LENGTH_SIGNAL
       
        zh = np.multiply(inputs,self.W_h[j])
        result = mean(result)*diff_sigmoid(mean(zh))*inputs/zh.size
        
        return result


class atomic_formula():
    def __init__(self, encoder, decoder,  operator):
        self.encoder = encoder
        self.decoder = decoder
        self.operator = operator
        self.tau = None

    def set_init(self, encoder, decoder, operator):
        self.encoder = encoder
        self.decoder = decoder
        self.operator 

    def output(self, inputs):
        tau = self.encoder.output(inputs)
        self.tau = tau 

        weight  = self.decoder.output(np.array([tau[0],tau[1]]))
        self.operator.set_bias(np.array(tau[2]))
        self.operator.set_rho(inputs)
        self.operator.set_weight(weight)
        self.operator.set_interval(int(tau[0]),int(tau[1]))

        return self.operator.robustness(weight,inputs)

            
class DAE():

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
            

    def set_init(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


    def output(self, inputs):
        h_out = self.encoder.output(inputs)
        signal  = self.decoder.output(h_out)
        return signal




class feedforward():

    def __init__(self, W_o, neurons):
        self.W_o = W_o 
        self.neurons = neurons

    def set_weight(self, W_o):
        self.W_o = W_o
     
    def output(self, rho):
        out = []
        for w, neuron in zip(self.W_o, self.neurons):
            z = neuron.robustness(w, rho)
            out.append(z)
        return  np.array(out)


    def gradient_w(self, i):
        # i the ith neuron in output layer
        neuron = self.neurons[i]
        result= neuron.gradient_w()
        return result

    def gradient_r(self,  inputs, Grad_r):
        #ith neuron in input
        result =0.0 
        for w, neuron, grad_r in zip(self.W_o, self.neurons, Grad_r):
            neuron.set_weight(w)
            neuron.set_rho(inputs)
            result = result+ grad_r*neuron.gradient_r()
        return result 






   








        





