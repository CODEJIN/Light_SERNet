import matplotlib
import torch, argparse, yaml, os, logging
from Modules import Generator, Variance_Predictors
from Arg_Parser import Recursive_Parse

import librosa
import matplotlib.pyplot as plt

class Tracer(torch.nn.Module):
    def __init__(
        self,
        generator: Generator,
        variance_predictors: Variance_Predictors,
        vocoder: torch.nn.Module
        ):
        super().__init__()
        self.generator = generator
        self.variance_predictors = variance_predictors
        self.vocoder = vocoder

    def forward(self, audios: torch.Tensor, speakers: torch.Tensor):
        features = self.generator.mel_generator(audios.unsqueeze(0))
        log_f0s, energies = self.variance_predictors(features)

        conversions = self.generator(
            sources= features,
            speakers= speakers,
            log_f0s= log_f0s,
            energies= energies
            )
        conversions = (conversions.clamp(-1.0, 1.0) + 1.0) / 2.0 * (2.0957 + 11.5129) - 11.5129
        conversion_audios = self.vocoder(conversions)

        return conversion_audios


@torch.no_grad()
def Trace(
    checkpoint_path: str,
    output_path: str,
    hyper_parameters: argparse.Namespace
    ):
    generator = Generator(hyper_parameters= hyper_parameters)
    variance_predictors = Variance_Predictors(hyper_parameters= hyper_parameters)
    state_dict = torch.load(checkpoint_path, map_location= 'cpu')    
    generator.load_state_dict(state_dict['Model']['Generator'])
    variance_predictors.load_state_dict(state_dict['Model']['Variance_Predictors'])
    generator.eval()
    variance_predictors.eval()

    vocoder = torch.jit.load('vocgan_jit_gta_jdit_ganspeech_2000.pts', map_location='cpu')
    vocoder.eval()
    logging.info('Model loaded.')

    tracer = Tracer(
        generator= generator,
        variance_predictors= variance_predictors,
        vocoder= vocoder
        )
    # audio = torch.randn(16384)
    # speaker = torch.randint(0, hyper_parameters.Speakers, size=(1,))
    
    audio = librosa.load('C:/Users/Heejo.You/Desktop/test_sound/SGEAI_JDJ.wav', sr=22050)[0]
    audio = librosa.effects.trim(audio, frame_length= 512, hop_length= 256)[0]
    audio = librosa.util.normalize(audio) * 0.95
    audio = torch.from_numpy(audio)
    speakers = torch.randint(0, 24, size=(1,))

    traced_generator = torch.jit.trace(
        tracer,
        (audio, speakers)
        )
    logging.info('Tracer generated.')
    if os.path.dirname(output_path) != '':
        os.makedirs(os.path.dirname(output_path), exist_ok= True)
    traced_generator.save(output_path)
    logging.info('Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-c', '--checkpoint_path', required= True, type= str)
    parser.add_argument('-o', '--out_path', required= True, type= str)
    args = parser.parse_args()

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    Trace(
        checkpoint_path= args.checkpoint_path,
        output_path= args.out_path,
        hyper_parameters= hp
        )
    
# python Trace.py -hp Hyper_Parameters.yaml -c S_50000.pt -o trace.pts