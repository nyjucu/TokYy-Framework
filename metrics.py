import torch
from tokyy.utils import LogType, log_message
from enum import Enum
from typing import Callable, List, Dict
import numpy as np

class Metrics( str, Enum ):
    MAE = 'mae'
    MSE = 'mse'

    ABS_REL = 'abs_rel'
    SQ_REL = 'sq_rel'
    
    RMSE = 'rmse'
    RMSE_LOG = 'rmse_log'
    
    DELTA1 = 'delta1'
    DELTA2 = 'delta2'
    DELTA3 = 'delta3'


def mae( y_pred, y_true ) -> float:
    return torch.mean( torch.abs( y_pred - y_true ) )


def mse( y_pred, y_true ) -> float:
    return torch.mean( torch.square( y_pred - y_true ) )


def abs_rel( y_pred, y_true, epsilon = 1e-6 ) -> float:
    y_true_safe = y_true + epsilon
    return torch.mean( torch.abs( y_pred - y_true ) / y_true_safe )


def sq_rel( y_pred, y_true, epsilon = 1e-6 ) -> float:
    y_true_safe = y_true + epsilon
    return torch.mean( torch.square( y_pred - y_true ) / y_true_safe )


def rmse( y_pred, y_true ) -> float:
    return torch.sqrt( torch.mean( ( y_pred - y_true ) ** 2 ) )


def rmse_log( y_pred, y_true ) -> float:
    y_pred_safe = y_pred + 1e-6
    y_true_safe = y_true + 1e-6

    return torch.sqrt( ( ( torch.log( y_pred_safe ) - torch.log( y_true_safe ) ) ** 2 ).mean() )


def delta1( y_pred, y_true ) -> float:
    thresh = torch.max(y_pred / y_true, y_true / y_pred )
    return torch.mean( ( thresh < 1.25 ).float() )


def delta2( y_pred, y_true ) -> float:
    thresh = torch.max( y_pred / y_true, y_true / y_pred )
    return torch.mean( ( thresh < 1.5625 ).float() )


def delta3( y_pred, y_true ) -> float:
    thresh = torch.max( y_pred / y_true, y_true / y_pred )
    return torch.mean( ( thresh < 1.953125 ).float() )


metric_function: Dict[ Metrics, Callable ] = {
    Metrics.MAE : mae,
    Metrics.MSE : mse,
    
    Metrics.ABS_REL : abs_rel,
    Metrics.SQ_REL : sq_rel,
    
    Metrics.RMSE : rmse,
    Metrics.RMSE_LOG : rmse_log,

    Metrics.DELTA1 : delta1,
    Metrics.DELTA2 : delta2,
    Metrics.DELTA3 : delta3,
}


class Metric:
    device = torch.device( 'cpu' )
        
    def __init__( self, metrics : List[ Metrics ], device : torch.device = None ):
        self.metrics = metrics
        self.device = device
        self.computed = {}

    def compute_metrics( self, y_pred, y_true, batch_size = 1 ) -> Dict[ Metrics, float ]:
        y_pred = y_pred.to( self.device )
        y_true = y_true.to( self.device )
        batch_size = float( batch_size )

        self.computed = {}

        for metric in self.metrics:
            self.computed[ metric ] = metric_function[ metric ]( y_pred, y_true ) / batch_size

        return self.computed
    
    def set_device( self, device_name = 'cpu' ):
        if device_name == 'cpu':
            Metric.device = torch.device( 'cpu' )

            return log_message( LogType.SUCCESS, "Metric's device changed to CPU" )

        elif device_name == 'cuda':
            if torch.cuda.is_available():
                Metric.device = torch.device( 'cuda' )

                log_message( LogType.SUCCESS, "Metric's changed to CUDA" )
                log_message( LogType.NONE, f" Current device number: { torch.cuda.current_device() }" )
                log_message( LogType.NONE, f" Current device name: { torch.cuda.get_device_name() }" )
                return LogType.SUCCESS.value            
            
            else:
                return log_message( LogType.WARNING, "CUDA is not available. Device remains CPU" )
    
        else:
            return log_message( LogType.WARNING, f"Provided device name not \'cpu\' or \'cuda\'. Device remains { self.device } ")
