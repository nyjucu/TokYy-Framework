from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import torch
import os
import h5py
import numpy as np

def rgb_transform( rgb_np, size = ( 128, 128 ) ):
    rgb_img = Image.fromarray( np.transpose( rgb_np, ( 1, 2, 0 ) ).astype( np.uint8 ) )
    
    transform_pipeline = T.Compose( [
        T.Resize( size ),
        T.ToTensor(), 
    ] )
    
    return transform_pipeline( rgb_img )


def depth_transform( depth_np, size = ( 128, 128 ) ):
    depth_img = Image.fromarray( depth_np.astype( np.float32 ) )
    
    transform_pipeline = T.Compose( [
        T.Resize( size ),
        T.ToTensor()
    ])
    
    return transform_pipeline( depth_img )


class Dataset( Dataset ):
    def __init__( self, root_dir : str, size : tuple = ( 128, 128 ), transform = rgb_transform, depth_transform = depth_transform ):
        self.samples = []
        self.transform = transform
        self.depth_transform = depth_transform
        self.size = size

        has_subdirs = any( os.path.isdir( os.path.join( root_dir, d )) for d in os.listdir( root_dir ) )

        if has_subdirs:
            for scene in os.listdir( root_dir ):
                scene_dir = os.path.join( root_dir, scene )
                if not os.path.isdir( scene_dir ):
                    continue
                for fname in os.listdir( scene_dir ):
                    if fname.endswith( '.h5' ):
                        self.samples.append( os.path.join( scene_dir, fname ) )
        else:
            for fname in os.listdir( root_dir ):
                if fname.endswith( '.h5' ):
                    self.samples.append( os.path.join( root_dir, fname ) )

    def __len__( self ):
        return len( self.samples )

    def __getitem__( self, idx ):
        file_path = self.samples[ idx ]

        with h5py.File( file_path, 'r' ) as f:
            rgb = np.array( f[ 'rgb' ] )
            depth = np.array( f[ 'depth' ] )

        if self.transform:
            rgb = self.transform( rgb, self.size )
        else:
            rgb = torch.from_numpy( rgb ).float() / 255.0

        if self.depth_transform:
            depth = self.depth_transform( depth, self.size )
        else:
            depth = torch.from_numpy( depth ).float().unsqueeze( 0 )

        return rgb, depth
