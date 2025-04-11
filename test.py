import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SparseAttnlogic import ResEventually
class Conv1DEventuallyLayer(nn.Module):
    def __init__(self, channels, kernel_size, max_limit=1e6):
        super(Conv1DEventuallyLayer, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.max_limit = max_limit
        # Initialize weights as a trainable parameter
        # Shape: [channels, kernel_size] assuming weights are applied per-channel
        self.weights = nn.Parameter(torch.randn(channels, kernel_size))

    def eventually(self, wrho):
        # Assume wrho shape: [batch_size, channels, window_size]
        wrho = wrho.double()
        neg_wrho = torch.where(wrho <= 0.0, 1 - wrho, 1.0)
        neg_prod = torch.prod(neg_wrho, -1)
        neg_prod = neg_prod.double()
        neg_num = torch.where(torch.abs(wrho) > 0, 1.0, 0.0)
        neg_prod = torch.where(neg_prod < self.max_limit, neg_prod, self.max_limit * torch.sigmoid(neg_prod))
        power = torch.sum(neg_num, -1) + 0.000001
        neg_result = -neg_prod ** (1 / power) + 1
        pos_wrho = torch.where(wrho > 0, wrho, 0.0)
        pos_sum = torch.sum(pos_wrho, -1)
        pos_result = pos_sum / power
        result = torch.where(pos_result > 0, pos_result, neg_result)
        return result

    def forward(self, x):
        # Unfold x to create windows
        # Initial windows shape: [batch_size, channels, kernel_size, num_windows]
        windows = x.unfold(dimension=2, size=self.kernel_size, step=1)
        # Permute windows to shape [batch_size, channels, num_windows, kernel_size]
        # to align with weights_adjusted for broadcasting
        windows_permuted = windows.permute(0, 1, 3, 2)
        # weights_adjusted shape: [1, channels, kernel_size, 1] for broadcasting
        weights_adjusted = self.weights.unsqueeze(0).unsqueeze(-1)
        # Perform element-wise multiplication
        # This now correctly aligns windows_permuted with weights_adjusted for broadcasting
        weighted_windows = windows_permuted * weights_adjusted

        # Since the shape of weighted_windows is now [batch_size, channels, num_windows, kernel_size],
        # and you may want to apply 'eventually' across the last dimension,
        # you should reshape or permute as necessary before applying 'eventually'
        weighted_windows_reshaped = weighted_windows.permute(0, 1, 3, 2).contiguous()
        results = self.eventually(weighted_windows_reshaped)
        return results




 


res = ResEventually(256)

x = torch.randn(20,256)
y = torch.randn(20,256)
output = res(x,y)
print(output.size())