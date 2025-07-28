import torch
import os
from tokyy.utils import log_message, LogType
from tokyy.metrics import Metrics, Metric


class Checkpointer:
    def __init__( self, model, optimizer, scaler = None, scheduler = None ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.epoch = 0

        self.losses = {
            'train_loss' : [],
            'val_loss' : []
        }

        self.metrics = {}
        for metric in Metrics:
            self.metrics[ metric ] = []

    def save( self, path ):
        checkpoint = {}

        checkpoint[ 'model' ] = self.model.state_dict()
        checkpoint[ 'optimizer' ] = self.optimizer.state_dict()

        if self.scaler:
            checkpoint[ 'scaler' ] = self.scaler.state_dict()

        if self.scheduler:
            checkpoint[ 'scheduler' ] = self.scheduler.state_dict()

        checkpoint[ 'epoch' ] = self.epoch
        checkpoint[ 'losses' ] = self.losses
        checkpoint[ 'metrics' ] = self.metrics

        try:             
            print( path )

            if os.path.exists( path ):
                log_message( LogType.WARNING, f"Overwriting to checkpoint saving path { path }" )

            torch.save( checkpoint, path )
            return log_message( LogType.SUCCESS, f"Checkpoint saved at epoch { self.epoch + 1 } to { path }" )
        
        except: 
            return log_message( LogType.ERROR, f"Checkpoint failed saving" )


    def update( self, model, optimizer, epoch, scaler = None, scheduler = None, losses = {}, metrics = {} ):
        self.model = model
        self.optimizer = optimizer

        if scaler:
            self.scaler = scaler
        
        if scheduler:
            self.scheduler = scheduler

        self.epoch = epoch

        for key, val in losses.items():
            if key not in self.losses:
                log_message( LogType.WARNING, f"Provided loss { key } is not accepted")
                continue
            
            self.losses[ key ].append( val )
            
        for key, val in metrics.items():
            if key not in self.metrics:
                log_message( LogType.WARNING, f"Provided metric { key } is not accepted")
                continue
            
            self.metrics[ key ].append( val )

        log_message( LogType.SUCCESS, "Checkpoint updated" )

    @classmethod
    def load( cls, path, model, optimizer, scaler = None, scheduler = None, map_loc = 'cuda' ):
        if not os.path.exists(path):
            log_message( LogType.ERROR, "Checkpoint load path doesn't exist.")
            return None
    
        checkpoint = torch.load( path, map_location = map_loc, weights_only = False)
        
        instance = cls( model, optimizer, scaler, scheduler )

        if 'model' in checkpoint:
            instance.model.load_state_dict( checkpoint[ 'model' ] )
        else:
            return log_message( LogType.ERROR, "Model not found in checkpoint" )

        if 'optimizer' in checkpoint:
            instance.optimizer.load_state_dict( checkpoint[ 'optimizer' ] )
        else:
            log_message( LogType.WARNING, "Optimizer not found in checkpoint" )
        
        if scaler and 'scaler' in checkpoint:
            instance.scaler.load_state_dict( checkpoint[ 'scaler' ] )
        else:
            log_message( LogType.WARNING, "Scaler not found in checkpoint" )
        
        if scheduler and 'scheduler' in checkpoint:
            instance.scheduler.load_state_dict( checkpoint[ 'scheduler' ] )
        else:
            log_message( LogType.WARNING, "Scheduler not found in checkpoint" )
        
        if 'losses' in checkpoint:
            instance.losses = checkpoint[ 'losses' ]
        else:
            log_message( LogType.WARNING, "Losses not found in checkpoint" )
        
        if 'metrics' in checkpoint:
            instance.metrics = checkpoint[ 'metrics' ]
        else:
            log_message( LogType.WARNING, "Metrics not found in checkpoint" )

        log_message( LogType.SUCCESS, "Checkpoint loaded" )

        return instance
    
    @staticmethod
    def get_epoch( path, map_loc = 'cpu' ):
        if not os.path.exists( path ):
            log_message( LogType.WARNING, "Checkpoint path doesn't exist. Loaded epoch 0" )
            return 0

        checkpoint = torch.load( path, map_location = map_loc, weights_only = False )
        epoch = checkpoint.get( 'epoch', 0 )

        log_message( LogType.SUCCESS, f"Checkpointer loaded epoch { epoch + 1 }" )

        return epoch


    def show_metrics( self, only_last = True ):
        print( self.metrics.items() )
        for key, val in self.metrics.items():
            if len( val ) > 0:
                print( key.value, end = ' ' )
                
                if only_last: print( val[ - 1 ] )
                else: print( val )


    def show_losses( self, only_last = True  ):
        for key, val in self.losses.items():
            if len( val ) > 0:
                print( key, end = ' ' )
                
                if only_last: print( val[ - 1 ] )
                else: print( val )
