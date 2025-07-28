from tokyy.utils import LogType, log_message, ask_yes_no
from tokyy.checkpointer import Checkpointer
from tokyy.metrics import Metric, Metrics

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

import os

from tqdm import tqdm

from typing import List

class Trainer():
    ask_before = True
    device = torch.device( 'cpu' )
    non_blocking = True

    def __init__( self, model : torch.nn, optimizer : torch.optim, criterion : torch.nn, metrics : List[ Metrics ], checkpoint_path : str, scaler = None, scheduler = None ):
        self.model = model.to( Trainer.device )
        self.optimizer = optimizer
        self.criterion = criterion.to( Trainer.device )
        self.metric = Metric( metrics = metrics, device = Trainer.device )
        self.scaler = scaler
        self.scheduler = scheduler  
        self.checkpointer = Checkpointer( model, optimizer, scaler = scaler, scheduler = scheduler )
        self.checkpoint_path = checkpoint_path

        self.num_workers = 8

        self.dataset = None
        self.train_loader = None
        self.val_loader = None

        self.accum_steps = 2
        self.batch_size = 32
        self.max_epochs = 100
        self.epochs_per_session = 20

        log_message( LogType.NONE, f"Using device: { self.device }" )


    def set_device( self, device_name = 'cpu' ):
        if device_name == 'cpu':
            Trainer.device = torch.device( 'cpu' )

            return log_message( LogType.SUCCESS, "Trainer's device changed to CPU" )

        elif device_name == 'cuda':
            if torch.cuda.is_available():
                Trainer.device = torch.device( 'cuda' )

                self.model = self.model.to( Trainer.device )
                self.criterion = self.criterion.to( Trainer.device )
                self.metric.device = Trainer.device

                log_message( LogType.SUCCESS, "Trainer's device changed to CUDA" )
                log_message( LogType.NONE, f" Current device number: { torch.cuda.current_device() }" )
                log_message( LogType.NONE, f" Current device name: { torch.cuda.get_device_name() }" )
                return LogType.SUCCESS.value            
            
            else:
                return log_message( LogType.WARNING, "CUDA is not available. Device remains CPU" )
    
        else:
            return log_message( LogType.WARNING, f"Provided device name not \'cpu\' or \'cuda\'. Device remains { self.device } ")


    def set_model( self, model ):
        self.model = model.to( self.device )


    def set_criterion( self, criterion ):
        self.criterion = criterion.to( self.device )


    def load_dataset( self, dataset, train_path, size, val_path = None ):
        if not os.path.exists( train_path ):
            return log_message( LogType.ERROR, "Training dataset path does not exist. Qutting..." )

        if val_path is not None and not os.path.exists( val_path ):
            return log_message( LogType.ERROR, "Validation dataset path does not exist. Quitting..." )

        can_begin = ask_yes_no( "Load dataset?" ) if Trainer.ask_before else True

        if not can_begin:
            return log_message( LogType.NONE, "User input didn't allow dataset loading. Quittting...")

        log_message( LogType.NONE, "Dataset started loading")

        torch.manual_seed( 69 )
        
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all( 69 )

        train_dataset = dataset( root_dir = train_path, size = size )
        val_dataset = dataset( root_dir = val_path, size = size )

        log_message( LogType.SUCCESS, "Datasets loaded" )

        self.train_loader = DataLoader( train_dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.num_workers, pin_memory = True )
        self.val_loader = DataLoader( val_dataset, batch_size = self.batch_size, shuffle = False, num_workers = self.num_workers, pin_memory = True )

        log_message( LogType.NONE, f"Loaded { len( self.train_loader.dataset ) } training samples" )
        log_message( LogType.NONE, f"Loaded { len( self.val_loader.dataset ) } training samples" )


    def train_supervised( self ):
        can_begin = ask_yes_no( "Begin training?" ) if Trainer.ask_before else True

        if not can_begin:
            return log_message( LogType.NONE, "User input didn't allow training. Quittting...")

        if self.train_loader is None:
            return log_message( LogType.ERROR, "Training loader is empty. Quitting... " )

        start_epoch = self.checkpointer.get_epoch( self.checkpoint_path )

        for epoch in range( start_epoch, min( start_epoch + self.epochs_per_session, self.max_epochs ) ):
            self.model.train()

            train_loss = torch.tensor( 0.0, device = self.device )
            self.optimizer.zero_grad()

            loop = tqdm( enumerate( self.train_loader ), total = len( self.train_loader ), desc = f"Epoch { epoch + 1 } / { self.max_epochs }" )

            for i, ( input_data, target ) in loop:
                input_data = input_data.to( self.device, non_blocking = Trainer.non_blocking )
                target = target.to( self.device, non_blocking = Trainer.non_blocking )

                with autocast( device_type = self.device.type ):
                    output_data = self.model( input_data )
                    loss = self.criterion( output_data, target ) / self.accum_steps

                if torch.isfinite( loss) :
                    self.scaler.scale( loss ).backward()
                else:
                    log_message( LogType.WARNING, f"Skipping batch { i } due to invalid loss: { loss.item() }" )
                    continue

                if ( i + 1 ) % self.accum_steps == 0 or ( i + 1 ) == len( self.train_loader ):
                    self.scaler.step( self.optimizer )
                    self.scaler.update()
                    self.optimizer.zero_grad()

                train_loss += loss.detach() * input_data.size( 0 )

                loop.set_postfix( loss = loss.item() )

            log_message( LogType.NONE, "Training completed. Evaluating on validation set..." )

            self.model.eval()

            val_loss = torch.tensor( 0.0, device = self.device )

            with torch.no_grad():
                for input_data, target in self.val_loader:
                    input_data = input_data.to( self.device, non_blocking = Trainer.non_blocking )
                    target = target.to( self.device, non_blocking = Trainer.non_blocking )

                    with autocast( device_type = self.device.type ):
                        output_data = self.model( input_data )
                        loss = self.criterion( output_data, target )

                        self.metric.compute_metrics( output_data, target, batch_size = len( self.val_loader.dataset ) )

                        val_loss += loss.detach() * input_data.size( 0 )

            train_loss = train_loss.item()
            val_loss = val_loss.item()

            losses = {
                'train_loss' : train_loss / len( self.train_loader.dataset ),
                'val_loss' : val_loss / len( self.val_loader.dataset )
            }

            log_message( LogType.NONE, f"Epoch [ {epoch + 1} / { self.max_epochs } ], Train Loss: {losses[ 'train_loss' ]:.4f}, Val Loss: {losses[ 'val_loss']:.4f}")

            self.checkpointer.update(
                model = self.model,
                optimizer = self.optimizer,
                scaler = self.scaler,
                scheduler = self.scheduler,

                epoch = epoch,

                losses = losses,
                metrics = self.metric.computed
            )

            self.checkpointer.save( self.checkpoint_path )

            torch.cuda.empty_cache()
            
            log_message( LogType.SUCCESS, f"Model trained for { epoch + 1 } epoch[s]" )