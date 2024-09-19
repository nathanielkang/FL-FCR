import torch
from conf import conf
from collections import Counter
import torch.nn.functional as F
class FedLCalibratedLoss():
    def __init__(self, tau=conf['--tau']):
        """
        Initializes the FedLCalibratedLoss.

        Args:
            tau (float): The tau parameter for calibration.
        """
        self.tau = tau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Assuming `conf` is a dictionary containing configuration parameters
        self.num_classes = conf.get('num_classes', 0)  # Get the number of classes from the config
        if self.num_classes == 0:
            raise ValueError("Number of classes not found in configuration.")

        self.label_distrib = torch.zeros(self.num_classes, device=self.device)

    def logit_calibrated_loss(self,logit, y):
        """
        Computes the loss for a given batch of logits and labels.

        Args:
            logit (torch.Tensor): The logits predicted by the model.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        label_counter = Counter(y.tolist()) # count number of label occurance in batch
        for cls, count in label_counter.items():
            self.label_distrib[cls] = max(1e-8, count)

        cal_logit = torch.exp(
            logit
            - (
                self.tau
                * torch.pow(self.label_distrib, -1 / 4)
                .unsqueeze(0)
                .expand((logit.shape[0], -1))
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.sum() / logit.shape[0]

# v2 for faster processing
class FedLCalibratedLoss():
    def __init__(self, tau=conf['--tau']):
        """
        Initializes the FedLCalibratedLoss.

        Args:
            tau (float): The tau parameter for calibration.
        """
        self.tau = tau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = conf.get('num_classes', 0)  # Get the number of classes from the config
        
        if self.num_classes == 0:
            raise ValueError("Number of classes not found in configuration.")
        
        # Initialize label distribution tensor
        self.label_distrib = torch.zeros(self.num_classes, device=self.device)

    def logit_calibrated_loss(self, logit, y):
        """
        Computes the loss for a given batch of logits and labels.

        Args:
            logit (torch.Tensor): The logits predicted by the model.
            y (torch.Tensor): The true labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Calculate label distribution using tensor operations for better performance
        batch_size = y.size(0)
        label_counts = torch.bincount(y, minlength=self.num_classes).float()
        
        # Ensure there are no zero counts for stability
        self.label_distrib = torch.maximum(label_counts, torch.tensor(1e-8, device=self.device))

        # Compute calibration term only once
        calibration_term = torch.pow(self.label_distrib, -1 / 4).unsqueeze(0)


        # Calibrate the logits
        cal_logit = torch.exp(logit - (self.tau * calibration_term))
        
        # Gather logits corresponding to the true labels
        y_logit = torch.gather(cal_logit, dim=-1, index=y.unsqueeze(1))
        
        # Compute the calibrated loss
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
    

        # Return the mean loss over the batch
        return loss.mean()

