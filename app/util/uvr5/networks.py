"""
UVR5 Neural Network Architectures for Audio Separation.

Contains two model architectures:
- CascadedASPPNet: For HP5 vocal isolation models
- CascadedNet: For DeEcho/DeReverb models (uses LSTM)

From Mangio-RVC-Fork/lib/uvr5_pack/lib_v5/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import spec_utils


# ==============================================================================
# Shared Utility Functions
# ==============================================================================

def crop_center(h1, h2):
    """Crop h1 to match h2's time dimension."""
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    return h1[:, :, :, s_time:e_time]


# ==============================================================================
# HP5 Model Layers (CascadedASPPNet)
# ==============================================================================

class Conv2DBNActiv(nn.Module):
    """Convolution + BatchNorm + Activation."""
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin, nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)


class SeperableConv2DBNActiv(nn.Module):
    """Separable Convolution + BatchNorm + Activation (for HP5)."""
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=nn.ReLU):
        super(SeperableConv2DBNActiv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                nin, nin,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=nin,
                bias=False,
            ),
            nn.Conv2d(nin, nout, kernel_size=1, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def __call__(self, x):
        return self.conv(x)


class HP5Encoder(nn.Module):
    """Encoder for HP5 models - returns (output, skip_connection)."""
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(HP5Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, stride, pad, activ=activ)

    def __call__(self, x):
        skip = self.conv1(x)
        h = self.conv2(skip)
        return h, skip


class HP5Decoder(nn.Module):
    """Decoder for HP5 models."""
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super(HP5Decoder, self).__init__()
        self.conv = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class HP5ASPPModule(nn.Module):
    """ASPP Module for HP5 models (uses separable convolution)."""
    def __init__(self, nin, nout, dilations=(4, 8, 16), activ=nn.ReLU):
        super(HP5ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, nin, 1, 1, 0, activ=activ)
        self.conv3 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = SeperableConv2DBNActiv(nin, nin, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = nn.Sequential(
            Conv2DBNActiv(nin * 5, nout, 1, 1, 0, activ=activ),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode="bilinear", align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        return self.bottleneck(out)


class BaseASPPNet(nn.Module):
    """Base ASPP network for HP5 models."""
    def __init__(self, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.enc1 = HP5Encoder(nin, ch, 3, 2, 1)
        self.enc2 = HP5Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = HP5Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = HP5Encoder(ch * 4, ch * 8, 3, 2, 1)
        self.aspp = HP5ASPPModule(ch * 8, ch * 16, dilations)
        self.dec4 = HP5Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = HP5Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = HP5Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = HP5Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def __call__(self, x):
        h, e1 = self.enc1(x)
        h, e2 = self.enc2(h)
        h, e3 = self.enc3(h)
        h, e4 = self.enc4(h)
        h = self.aspp(h)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = self.dec1(h, e1)
        return h


class CascadedASPPNet(nn.Module):
    """
    Network architecture for HP5 vocal isolation models.
    Uses cascaded ASPP networks with separable convolutions.
    """
    
    def __init__(self, n_fft):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(2, 32)
        self.stg1_high_band_net = BaseASPPNet(2, 32)
        self.stg2_bridge = Conv2DBNActiv(34, 16, 1, 1, 0)
        self.stg2_full_band_net = BaseASPPNet(16, 32)
        self.stg3_bridge = Conv2DBNActiv(66, 32, 1, 1, 0)
        self.stg3_full_band_net = BaseASPPNet(32, 64)
        self.out = nn.Conv2d(64, 2, 1, bias=False)
        self.aux1_out = nn.Conv2d(32, 2, 1, bias=False)
        self.aux2_out = nn.Conv2d(32, 2, 1, bias=False)
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.offset = 128

    def forward(self, x, aggressiveness=None):
        mix = x.detach()
        x = x.clone()
        x = x[:, :, : self.max_bin]
        bandw = x.size()[2] // 2
        aux1 = torch.cat([
            self.stg1_low_band_net(x[:, :, :bandw]),
            self.stg1_high_band_net(x[:, :, bandw:]),
        ], dim=2)
        h = torch.cat([x, aux1], dim=1)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(h))
        h = torch.cat([x, aux1, aux2], dim=1)
        h = self.stg3_full_band_net(self.stg3_bridge(h))
        mask = torch.sigmoid(self.out(h))
        mask = F.pad(input=mask, pad=(0, 0, 0, self.output_bin - mask.size()[2]), mode="replicate")
        
        if self.training:
            aux1 = torch.sigmoid(self.aux1_out(aux1))
            aux1 = F.pad(input=aux1, pad=(0, 0, 0, self.output_bin - aux1.size()[2]), mode="replicate")
            aux2 = torch.sigmoid(self.aux2_out(aux2))
            aux2 = F.pad(input=aux2, pad=(0, 0, 0, self.output_bin - aux2.size()[2]), mode="replicate")
            return mask * mix, aux1 * mix, aux2 * mix
        else:
            if aggressiveness:
                mask[:, :, : aggressiveness["split_bin"]] = torch.pow(
                    mask[:, :, : aggressiveness["split_bin"]], 1 + aggressiveness["value"] / 3
                )
                mask[:, :, aggressiveness["split_bin"] :] = torch.pow(
                    mask[:, :, aggressiveness["split_bin"] :], 1 + aggressiveness["value"]
                )
            return mask * mix

    def predict(self, x_mag, aggressiveness=None):
        h = self.forward(x_mag, aggressiveness)
        if self.offset > 0:
            h = h[:, :, :, self.offset : -self.offset]
        assert h.size()[3] > 0
        return h


# ==============================================================================
# DeEcho/DeReverb Model Layers (CascadedNet)
# ==============================================================================

class DeEchoEncoder(nn.Module):
    """Encoder for DeEcho models - simpler, no skip return."""
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.LeakyReLU):
        super(DeEchoEncoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def __call__(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return h


class DeEchoDecoder(nn.Module):
    """Decoder for DeEcho models."""
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=nn.ReLU, dropout=False):
        super(DeEchoDecoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def __call__(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = spec_utils.crop_center(skip, x)
            x = torch.cat([x, skip], dim=1)
        h = self.conv1(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class DeEchoASPPModule(nn.Module):
    """ASPP Module for DeEcho models (standard convolution, configurable dropout)."""
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=nn.ReLU, dropout=False):
        super(DeEchoASPPModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ),
        )
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode="bilinear", align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.bottleneck(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LSTMModule(nn.Module):
    """LSTM module for DeEcho models - adds temporal context."""
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm),
            nn.BatchNorm1d(nin_lstm),
            nn.ReLU()
        )

    def forward(self, x):
        N, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0]  # N, nbins, nframes
        h = h.permute(2, 0, 1)  # nframes, N, nbins
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size()[-1]))  # nframes * N, nbins
        h = h.reshape(nframes, N, 1, nbins)
        h = h.permute(1, 2, 3, 0)
        return h


class BaseNet(nn.Module):
    """Base network for DeEcho models - includes LSTM."""
    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        self.enc1 = Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = DeEchoEncoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = DeEchoEncoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = DeEchoEncoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = DeEchoEncoder(nout * 6, nout * 8, 3, 2, 1)
        self.aspp = DeEchoASPPModule(nout * 8, nout * 8, dilations, dropout=True)
        self.dec4 = DeEchoDecoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = DeEchoDecoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = DeEchoDecoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = DeEchoDecoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def __call__(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        h = self.aspp(e5)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = torch.cat([h, self.lstm_dec2(h)], dim=1)
        h = self.dec1(h, e1)
        return h


class CascadedNet(nn.Module):
    """
    Network architecture for DeEcho and DeReverb UVR5 models.
    Uses LSTM modules for temporal context and cascaded BaseNet structure.
    """
    
    def __init__(self, n_fft, nout=32, nout_lstm=128):
        super(CascadedNet, self).__init__()
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64

        self.stg1_low_band_net = nn.Sequential(
            BaseNet(2, nout // 2, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0),
        )
        self.stg1_high_band_net = BaseNet(2, nout // 4, self.nin_lstm // 2, nout_lstm // 2)
        self.stg2_low_band_net = nn.Sequential(
            BaseNet(nout // 4 + 2, nout, self.nin_lstm // 2, nout_lstm),
            Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )
        self.stg2_high_band_net = BaseNet(nout // 4 + 2, nout // 2, self.nin_lstm // 2, nout_lstm // 2)
        self.stg3_full_band_net = BaseNet(3 * nout // 4 + 2, nout, self.nin_lstm, nout_lstm)
        self.out = nn.Conv2d(nout, 2, 1, bias=False)
        self.aux_out = nn.Conv2d(3 * nout // 4, 2, 1, bias=False)

    def forward(self, x):
        x = x[:, :, : self.max_bin]
        bandw = x.size()[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]

        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = torch.cat([l1, h1], dim=2)

        l2_in = torch.cat([l1_in, l1], dim=1)
        h2_in = torch.cat([h1_in, h1], dim=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = torch.cat([l2, h2], dim=2)

        f3_in = torch.cat([x, aux1, aux2], dim=1)
        f3 = self.stg3_full_band_net(f3_in)

        mask = torch.sigmoid(self.out(f3))
        mask = F.pad(input=mask, pad=(0, 0, 0, self.output_bin - mask.size()[2]), mode="replicate")

        if self.training:
            aux = torch.cat([aux1, aux2], dim=1)
            aux = torch.sigmoid(self.aux_out(aux))
            aux = F.pad(input=aux, pad=(0, 0, 0, self.output_bin - aux.size()[2]), mode="replicate")
            return mask, aux
        else:
            return mask

    def predict_mask(self, x):
        mask = self.forward(x)
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
        assert mask.size()[3] > 0
        return mask

    def predict(self, x, aggressiveness=None):
        mask = self.forward(x)
        pred_mag = x * mask
        if self.offset > 0:
            pred_mag = pred_mag[:, :, :, self.offset : -self.offset]
        assert pred_mag.size()[3] > 0
        return pred_mag

