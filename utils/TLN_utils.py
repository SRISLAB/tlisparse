# import the necessary packages
#from matplotlib.font_manager import _Weight
#from types import NoneType
from unittest import result
import numpy as np
from operator import xor
import torch
MAX_LIMIT =1000
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def basins_boxes(scores, tau=0.4, gamma=0.7):

    """
    Calculates the bounding boxes for a given signal (scores)
    scores: the scores of the points to be informative points
    tau: basin durations threshold
    gamma: water level
    """
    boxes  = bounding_boxes(scores,gamma)
    boxes  = merge_boxes(boxes,tau)
    if boxes is None:
        return np.zeros([1,4])
    else:
        return boxes


def merge_boxes(boxes, tau):
    # if there are no boxes, return an empty list
    if boxes is None:
        return None
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    nboxes = None
    idxs = np.argsort(x1)
    current = np.array([x1[idxs[0]],y1[idxs[0]],x2[idxs[0]],y2[idxs[0]]])

    for i in range(1,len(idxs)):
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        ii = idxs[i]
        width = x2[ii]-current[0]
        during =  x2[ii]-x1[ii]+current[2]-current[0]
        if during/width>tau:
            if nboxes is not None:
                nboxes=np.vstack((nboxes,np.array([current[0],current[1],x2[ii],y2[ii]])))
            else:
                nboxes = np.array([current[0],current[1],x2[ii],y2[ii]])
            current = np.array([current[0],current[1],x2[ii],y2[ii]])
        else:
            if nboxes is not None:
                nboxes=np.vstack((nboxes,current))
            else:
                nboxes =current
            current = boxes[ii,:]


    # return only the bounding boxes that were picked using the
    # integer data type

    return nboxes

def bounding_boxes(scores,gamma):
    boolean_scores = scores>gamma
 
    shift_right = np.append(boolean_scores.numpy(),[False])
    boolean_scores = np.insert(boolean_scores.numpy(),0,False)
    idx_bool = xor(boolean_scores,shift_right)
    idx_boole = idx_bool & boolean_scores
    idx_end = np.where(idx_boole==True)[0]
    idx_end[:]=[ele -1 for ele in idx_end]


    boolean_scores = scores>gamma
 
    shift_left = np.insert(boolean_scores.numpy(),0,False)
    boolean_scores = np.append(boolean_scores.numpy(),[False])
    idx_bool = xor(boolean_scores,shift_left)
    idx_bools  = idx_bool & boolean_scores
    idx_start = np.where(idx_bools==True)[0]
  

    assert len(idx_start) == len(idx_end)

    """     pick =[]
    for i in range(len(idx_start)):
        if idx_end[i]-idx_start[i]>1:
            pick.append(i)
    idx_start = idx_start[pick]
    idx_end   = idx_end[pick]
    """
    shape_start_end = [[start, end] for start, end in zip(idx_start, idx_end) if end>start+10]
    shape_s_e = list(zip(*shape_start_end))
    if len(shape_s_e)>0:
        boxes = np.zeros((len(shape_s_e[0]),4))
        boxes[:,0] = shape_s_e[0]
        boxes[:,2] = shape_s_e[1]
        boxes[:,3] = np.ones((1,len(shape_s_e[0])))   
    else:
        boxes = None 

    return boxes 


    
def selectweight(signal_len,box):

    select = np.ones((1,box[2]-box[0]))
    pre = np.zeros((1,box[0]-1))
    pro = np.zeros((1,signal_len-box[2]))

    weight = np.hstack((pre, select, pro))

    return weight 

class LogicOperator():
    def __init__(self, weight, robustness):
        self.weight = weight
        self.robustness = robustness

    def AND(self):
        if len(self.robustness)==0:
            return 0.0
        ewrho  = torch.mul(self.weight,self.robustness)
        if torch.min(ewrho)>=0:
            result = ewrho+1
            result = torch.prod(result)
            if result>MAX_LIMIT:
                result =MAX_LIMIT
            result = result**(1.0/len(self.robustness)) -1.0
        else:
            nrho = -ewrho
            nrho = torch.max(torch.zeros(len(nrho)),nrho)
            result = torch.sum(-nrho)
            result = result/len(self.robustness)

        return result


    def OR(self):
        if len(self.robustness)==0:
            return 0.0
        ewrho  = torch.mul(self.weight,self.robustness)
        if torch.max(ewrho)<0:
            result =1-ewrho
            result = torch.prod(result)
            if result>MAX_LIMIT:
                result =MAX_LIMIT
            result = -result**(1.0/len(self.robustness)) +1.0
        else:
            result = torch.sum(torch.max(torch.zeros(len(ewrho)),ewrho))
            result = result/len(self.robustness)
        return result


class TemporalOperator():
    def __init__(self, weight, robustness, box):
        self.weight = weight
        self.robustness = robustness
        self.tau1 = box[0]
        self.tau2 = box[2]

    def Always(self,shift=0):
        wrho  = torch.mul(self.weight,self.robustness)
        
        ewrho = wrho[min(self.tau1+shift,len(self.weight)-1):min(self.tau2+shift+1,len(self.weight)-1)]
        if len(ewrho)==0:
            return 0
        if torch.min(ewrho)>=0:
            result = ewrho+1
            result = torch.prod(result)
            if result>MAX_LIMIT:
                result = MAX_LIMIT
            result = result**(1.0/(self.tau2-self.tau1+1)) -1.0
        else:
            nrho = -ewrho
            nrho = torch.max(torch.zeros(len(nrho)),nrho)
            result = torch.sum(-nrho)
        
            result = result/(self.tau2-self.tau1+1)


        return result


    def Eventually(self,shift=0):
        wrho  = torch.mul(self.weight,self.robustness)
        ewrho = wrho[min(self.tau1+shift,len(self.weight)-1):min(self.tau2+shift+1,len(self.weight)-1)]
        
        if len(ewrho)==0:
            return 0

        if torch.max(ewrho)<0:
            result =1-ewrho
            result = torch.prod(result)
            if result>MAX_LIMIT:
                result  =MAX_LIMIT
            result = -result**(1.0/(self.tau2-self.tau1+1)) +1.0
        else:
            result = torch.sum(torch.max(torch.zeros(len(ewrho)),ewrho))
            result = result/(self.tau2-self.tau1+1)
     

        return result



    def AlwaysEventually(self,shift=0):

        erho =[]
        for i in range(0,shift+1):
            rob = self.Eventually(i)
            erho.append(rob)
        
        weight = torch.ones(len(erho))
        
        Logic = LogicOperator(weight,torch.tensor(erho))

        

        return Logic.AND()


    def EventuallyAlways(self,shift):

        erho =[]
        for i in range(0,shift+1):
            rob = self.Always(i)
            erho.append(rob)
        
        weight = torch.ones(len(erho))
        
        Logic = LogicOperator(weight,torch.tensor(erho))

        return Logic.OR()

    def output(self,shift):


        if self.tau2-self.tau1>0:

            return torch.tensor([self.Always(0),self.Eventually(0),self.AlwaysEventually(shift),self.EventuallyAlways(shift)])
        else:
            return torch.tensor([0,0,0,0])








