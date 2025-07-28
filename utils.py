from torchvision import transforms as T
from datetime import datetime
from enum import Enum
import argparse
import os


class LogType( Enum ):
    SUCCESS = 1
    ERROR = -1
    NONE = 0
    WARNING = 2

def ask_yes_no( question ):
    while True:
        answer = input( question + " [Y/N]: " ).strip().lower()
        if answer in ( 'y', 'yes', 'Y' ):
            return True
        elif answer in ( 'n', 'no', 'N' ):
            return False
        else:
            print( "Please enter Y or N." )


def log_message( type, msj = '' ):
    if type == LogType.ERROR:
        print( '\033[31m[  ERROR  ]\033[0m', end = ' ' )

    elif type == LogType.SUCCESS:
        print( '\033[32m[ SUCCESS ]\033[0m', end = ' ' )
    
    elif type == LogType.WARNING:
        print( '\033[33m[ WARNING ]\033[0m', end = ' ')

    print( '\033[34m', msj, '\033[0m', end = ' - ' )

    now = datetime.now()

    print( now.strftime( "%H:%M:%S" ) )

    return type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    script_dir = os.path.dirname( os.path.abspath( __file__ ) )
    checkpoint_dir = os.path.join( script_dir, "checkpoints" )
    os.makedirs( os.path.dirname( checkpoint_dir ), exist_ok = True )

    parser.add_argument( "--no_ask_before", action = "store_true", help = "Disable confirmation prompts in Trainer")
    parser.add_argument( "--checkpoint_dir", type = str, default = checkpoint_dir, help = "Path to the checkpoint directory. Default at /checkpoints" )
    parser.add_argument( "--checkpoint_name", type = str, default = "default.pt", help = "Checkpoint file name." )

    args = parser.parse_args()

    args.checkpoint_path = os.path.join( args.checkpoint_dir, args.checkpoint_name )

    log_message( LogType.NONE, f"Checkpoint absolute path is { args.checkpoint_path }" )

    if  not args.checkpoint_name.endswith( ( ".pt", ".pth", ".ckpt" ) ):
        log_message( LogType.WARNING, "It is recommended that the provided checkpoint file name ends with .pt, .pth, or .ckpt" )

    return args
