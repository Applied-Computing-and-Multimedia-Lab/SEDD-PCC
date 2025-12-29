import torch
import MinkowskiEngine as ME
from autoencoder import Encoder, Decoder_G, Decoder_A, get_ground_truth, get_coordinate
from autoencoder_PCGC import Encoder_PCGC, Decoder_PCGC
from functional import bound
from entropy_model import EntropyBottleneck
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UModel(torch.nn.Module):
    def __init__(self, hyper=True):
        super().__init__()
        self.hyper = hyper
        self.encoder = Encoder (channels=[3, 64, 128])
        self.decoder_g = Decoder_G (channels=[128, 64, 32, 16, 1])
        self.decoder_a = Decoder_A(channels=[3, 64, 128])
        self.get_coordinate = get_coordinate()
        self.get_ground_truth = get_ground_truth()
        self.entropy_bottleneck = EntropyBottleneck(128)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F,
            coordinate_map_key=data.coordinate_map_key,
            coordinate_manager=data.coordinate_manager,
            device=data.device)
        return data_Q, likelihood

    def forward(self, x, stage, training=True):
        if stage == 1:
            # Encoder
            y_list = self.encoder(x)
            y2 = y_list[0]
            ground_truth_list = y_list[1:] + [x]

            # Quantizer & Entropy Model
            y_q, likelihood = self.get_likelihood(y2,quantize_mode="noise" if training else "symbols")

            # Decoder_A
            out_cls_list_A, out_A, SM_A = self.decoder_a(y_q)

            F = bound(out_A.F, 0, 255)
            out0 = ME.SparseTensor(
                features=F,
                coordinate_map_key=out_A.coordinate_map_key,
                coordinate_manager=out_A.coordinate_manager,
                device=out_A.device)

            return {'out_A': out0,
                    'out_cls_list_A': out_cls_list_A,
                    'prior': y_q,
                    'likelihood': likelihood,
                    'ground_truth_list': ground_truth_list}

        if stage == 2:
            # Encoder
            y_list = self.encoder(x)
            y2 = y_list[0]
            ground_truth_list = y_list[1:] + [x]
            nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
                         for ground_truth in ground_truth_list]
            # Quantizer & Entropy Model
            y_q, likelihood = self.get_likelihood(y2,quantize_mode="noise" if training else "symbols")
            out_cls_list_G, out_g, SM_G = self.decoder_g(y_q, nums_list, ground_truth_list, training)

            return {'out_g': out_g,
                    'out_cls_list_G': out_cls_list_G,
                    'SM_G': SM_G,
                    'prior': y_q,
                    'likelihood': likelihood,
                    'ground_truth_list': ground_truth_list}

        if stage == 3:
            input = x
            y_list = self.encoder(x)
            y2 = y_list[0]
            ground_truth_list = y_list[1:] + [x]
            nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
                         for ground_truth in ground_truth_list]

            ### Quantizer & Entropy Model
            y_q, likelihood = self.get_likelihood(y2,quantize_mode="noise" if training else "symbols")

            # Decoder_G
            out_cls_list_G, out_g, SM_G = self.decoder_g(y_q, nums_list, ground_truth_list, training)

            ###get_coordinate
            out_gg = ME.SparseTensor(coordinates=out_g.C, features=out_g.F, device=out_g.device)


            coord1, coord2 = self.get_coordinate(out_gg)

            y_qa = ME.SparseTensor(
                features=y_q.F,
                coordinate_map_key=coord2.coordinate_map_key,
                coordinate_manager=coord2.coordinate_manager,
                device=x.device)

            # Decoder_A
            out_cls_list_A, out_A, SM_A = self.decoder_a(y_qa)



            F = bound(out_A.F, 0, 255)
            out0 = ME.SparseTensor(
                features=F,
                coordinate_map_key=out_A.coordinate_map_key,
                coordinate_manager=out_A.coordinate_manager,
                device=out_A.device)


            return {'out_A': out0,
                    'out_g': out_g,
                    'out_cls_list_G': out_cls_list_G,
                    'out_cls_list_A': out_cls_list_A,
                    'SM_G': SM_G,
                    'prior': y_q,
                    'likelihood': likelihood,
                    'ground_truth_list': ground_truth_list}



class PCGCModel(torch.nn.Module):
    def __init__(self, hyper=False):
        super().__init__()

        self.hyper = hyper

        self.encoder = Encoder_PCGC(channels=[1, 16, 32, 64, 32, 8])
        self.decoder = Decoder_PCGC(channels=[8, 64, 32, 16])
        self.entropy_bottleneck = EntropyBottleneck(8)

    def get_likelihood(self, data, quantize_mode):
        data_F, likelihood = self.entropy_bottleneck(data.F,
                                                     quantize_mode=quantize_mode)
        data_Q = ME.SparseTensor(
            features=data_F,
            coordinate_map_key=data.coordinate_map_key,
            coordinate_manager=data.coordinate_manager,
            device=data.device)

        return data_Q, likelihood

    def forward(self, x, training=True):
        # Encoder
        y_list = self.encoder(x)
        y = y_list[0]
        ground_truth_list = y_list[1:] + [x]
        nums_list = [[len(C) for C in ground_truth.decomposed_coordinates] \
            for ground_truth in ground_truth_list]

        # Quantizer & Entropy Model
        y_q, likelihood = self.get_likelihood(y,
            quantize_mode="noise" if training else "symbols")

        # Decoder
        out_cls_list, out = self.decoder(y_q, nums_list, ground_truth_list, training)

        return {'out':out,
                'out_cls_list':out_cls_list,
                'TM_G': y_q,
                'prior':y_q,
                'likelihood':likelihood,
                'ground_truth_list':ground_truth_list}

