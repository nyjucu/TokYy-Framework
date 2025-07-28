import torch
import torch.nn as nn


class ResidualConv( nn.Module ):
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
        
        self.l1 = ResidualConv( 3, 64 )
        self.l2 = ResidualConv( 64, 128 )
        self.l3 = ResidualConv( 128, 256 )
        self.l4 = ResidualConv( 256, 512 )
        self.l5 = ResidualConv( 512, 1024 )
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
        self.resconv = ResidualConv( out_channels + skip_channels, out_channels )

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
