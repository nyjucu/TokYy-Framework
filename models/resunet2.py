import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock( nn.Module ):
    def __init__( self, in_channels, out_channels ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( out_channels ),
            nn.ReLU( inplace = True ),
            nn.Conv2d( out_channels, out_channels, 3, padding = 1, stride = 1 ),
            nn.BatchNorm2d( out_channels )
        )

        self.skip = nn.Conv2d( in_channels, out_channels, 1, stride = 1 )

    def forward( self, x ):
        return torch.relu( self.conv( x ) + self.skip( x ) )
    

class SEBlock( nn.Module ):
    def __init__( self, channels, reduction_ratio = 16 ):
        super().__init__()

        self.reduced_dimension = max( 1, channels // reduction_ratio )

        self.mlp = nn.Sequential( 
            nn.Flatten(),
            nn.Linear( in_features = channels, out_features = self.reduced_dimension ),
            nn.ReLU( inplace = True ),
            nn.Linear( in_features = self.reduced_dimension, out_features = channels ),
            nn.Sigmoid()
        )

    def forward( self, x ):
        squeeze = F.adaptive_avg_pool2d( x, output_size = 1 )
        excitation = self.mlp( squeeze )

        return x * excitation.unsqueeze(-1).unsqueeze(-1)
    

class ChannelAttentionModule( nn.Module ):
    def __init__( self, channels, reduction_ratio = 16 ):
        super().__init__()

        self.reduced_dimension = max( 1, channels // reduction_ratio )

        self.mlp = nn.Sequential( 
            nn.Flatten(),
            nn.Linear( in_features = channels, out_features = self.reduced_dimension ),
            nn.ReLU( inplace = True ),
            nn.Linear( in_features = self.reduced_dimension, out_features = channels ),
        )

    def forward( self, x ):
        max_pool = F.adaptive_max_pool2d( x, output_size = 1 )
        avg_pool = F.adaptive_avg_pool2d( x, output_size = 1 )

        max_pool_after_mlp = self.mlp( max_pool )
        avg_pool_after_mlp = self.mlp( avg_pool )

        scale = torch.sigmoid(max_pool_after_mlp + avg_pool_after_mlp).unsqueeze(-1).unsqueeze(-1)

        return x * scale
    

class SpatialAttentionModule( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.conv = nn.Conv2d( in_channels = 2, out_channels = 1, kernel_size = 7, stride = 1, padding = 3, bias = False )

    def forward( self, x ):
        avg_pool = torch.mean( x, dim = 1, keepdim = True )      
        max_pool, _ = torch.max( x, dim = 1, keepdim = True )      

        pools = torch.cat( [ avg_pool, max_pool ], dim = 1 ) 
        pools_after_conv = self.conv( pools )

        scale = torch.sigmoid( pools_after_conv )

        return x * scale


class CBAMBlock( nn.Module ):
    def __init__( self, channels, reduction_ratio = 16 ):
        super().__init__()

        self.channel_am = ChannelAttentionModule( channels = channels, reduction_ratio = reduction_ratio )
        self.spatial_am = SpatialAttentionModule()

    def forward( self, x ):
        x_channel = self.channel_am( x )
        x_channel_spatial = self.spatial_am( x_channel )

        return x_channel_spatial
    

class ResSEBLock( nn.Module ):
    def __init__( self, in_channels, out_channels ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, 3, padding = 1, stride = 1, bias = False ),
            nn.BatchNorm2d( out_channels ),
            nn.ReLU( inplace = True ),
            nn.Conv2d( out_channels, out_channels, 3, padding = 1, stride = 1, bias = False ),
            nn.BatchNorm2d( out_channels ),
        )

        self.se = SEBlock( out_channels )

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d( in_channels, out_channels, 1, stride = 1, bias = False )

    def forward(self, x):
        x_conv = self.conv( x )
        x_se = self.se( x_conv )

        return torch.relu( x_se ) + self.skip( x )
    

class ResCBAMBlock( nn.Module ):
    def __init__( self, in_channels, out_channels ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d( in_channels, out_channels, 3, padding = 1, stride = 1, bias = False ),
            nn.BatchNorm2d( out_channels ),
            nn.ReLU( inplace = True ),
            nn.Conv2d( out_channels, out_channels, 3, padding = 1, stride = 1, bias = False ),
            nn.BatchNorm2d( out_channels ),
        )

        self.cbam = CBAMBlock( out_channels )

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d( in_channels, out_channels, 1, stride = 1, bias = False )

    def forward( self, x ):
        x_conv = self.conv( x )
        x_cbam = self.cbam( x_conv )

        return torch.relu( x_cbam ) + self.skip( x )


class AttentionGate( nn.Module ):
    def __init__( self, gate_channels, skip_channels, inter_channels ):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d( gate_channels, inter_channels, 1 ),
            nn.BatchNorm2d( inter_channels )
        )

        self.W_x = nn.Sequential(
            nn.Conv2d( skip_channels, inter_channels, 1 ),
            nn.BatchNorm2d( inter_channels )
        )

        self.psi = nn.Sequential(
            nn.Conv2d( inter_channels, 1, 1 ),
            nn.BatchNorm2d( 1 ),
            nn.Sigmoid()
        )

    def forward( self, g, x ):
        g1 = self.W_g( g )
        x1 = self.W_x( x )
        psi = self.psi( torch.relu( g1 + x1 ) )

        return x * psi


class Encoder( nn.Module ):
    def __init__( self ):
        super().__init__()
        
        self.l1 = ResCBAMBlock( 3, 64 )
        self.l2 = ResCBAMBlock( 64, 128 )
        self.l3 = ResCBAMBlock( 128, 256 )
        self.l4 = ResCBAMBlock( 256, 512 )
        self.l5 = ResCBAMBlock( 512, 1024 )
        self.pool = nn.MaxPool2d( 2, 2 )

    def forward( self, x ):
        x1 = self.l1( x )
        x2 = self.l2( self.pool( x1 ) )
        x3 = self.l3( self.pool( x2 ) )
        x4 = self.l4( self.pool( x3 ) )
        x5 = self.l5( self.pool( x4 ) )

        # x2 = self.l2( x1 )
        # x3 = self.l3( x2 )
        # x4 = self.l4( x3 )
        # x5 = self.l5( x4 )

        return x1, x2, x3, x4, x5


class DecoderBlock( nn.Module ):
    def __init__( self, in_channels, skip_channels, out_channels ):
        super().__init__()
        self.att = AttentionGate( skip_channels, skip_channels, skip_channels )
        self.upsamp = nn.ConvTranspose2d( in_channels, out_channels, 2, stride = 2 )
        self.resconv = ResBlock( out_channels + skip_channels, out_channels )

    def forward( self, x, skip ):
        x_up = self.upsamp( x )
        att_skip = self.att( x_up, skip )
        x_cat = torch.cat( [ x_up, att_skip ], dim = 1 ) 
        return self.resconv( x_cat )
        

class ResUNet( nn.Module ):
    def __init__( self ):
        super().__init__()

        self.encoder = Encoder()
        self.decoder1 = DecoderBlock( 1024, 512, 512 )
        self.decoder2 = DecoderBlock( 512, 256, 256 )
        self.decoder3 = DecoderBlock( 256, 128, 128 )
        self.decoder4 = DecoderBlock( 128, 64, 64 )
        self.output = nn.Conv2d( 64, 1, kernel_size = 1 )

    def forward( self, x ):
        x1, x2, x3, x4, x5 = self.encoder( x )
        d1 = self.decoder1( x5, x4 )
        d2 = self.decoder2( d1, x3 )
        d3 = self.decoder3( d2, x2 )
        d4 = self.decoder4( d3, x1 )
        out = self.output( d4 )

        return out
