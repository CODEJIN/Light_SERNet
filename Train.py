import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, pickle, wandb
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from librosa import griffinlim
from scipy.io import wavfile

from Modules import Light_SERNet
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater
from Radam import RAdam
from Noam_Scheduler import Modified_Noam_Scheduler
from Logger import Logger

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict


import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.gpu_id == 0:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }

            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp),
                    resume= self.steps > 0
                    )
                wandb.watch(self.model)


    def Dataset_Generate(self):
        emotion_info_dict = yaml.load(open(self.hp.Emotion_Info_Path), Loader=yaml.Loader)
                
        train_dataset = Dataset(
            emotion_info_dict= emotion_info_dict,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            feature_length_min= self.hp.Train.Train_Pattern.Feature_Length.Min,
            feature_length_max= self.hp.Train.Train_Pattern.Feature_Length.Max,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio
            )
        eval_dataset = Dataset(
            emotion_info_dict= emotion_info_dict,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            feature_length_min= self.hp.Train.Eval_Pattern.Feature_Length.Min,
            feature_length_max= self.hp.Train.Eval_Pattern.Feature_Length.Max,
            )
        inference_dataset = Inference_Dataset(
            n_fft= self.hp.Sound.N_FFT,
            num_features= self.hp.Sound.Mel_Dim,
            sample_rate= self.hp.Sound.Sample_Rate,
            frame_length= self.hp.Sound.Frame_Length,
            frame_shift= self.hp.Sound.Frame_Shift,
            fmin= self.hp.Sound.Mel_F_Min,
            fmax= self.hp.Sound.Mel_F_Max,
            source_paths= self.hp.Inference_Wav_Path,
            )

        if self.gpu_id == 0:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater()
        inference_collater = Inference_Collater()

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.DistributedSampler(eval_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model = Light_SERNet(self.hp).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon,
            weight_decay= self.hp.Train.Weight_Decay
            )    
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base= self.hp.Train.Learning_Rate.Base
            )        
        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

    def Train_Step(self, features, emotions):
        loss_dict = {}
        features = features.to(self.device, non_blocking=True)
        emotions = emotions.to(self.device, non_blocking=True)

        # Generator
        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            predictions = self.model(features)
            loss_dict['Classification'] = self.criterion(predictions, emotions)

        self.optimizer.zero_grad()
        self.scaler.scale(loss_dict['Classification']).backward()
        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for features, emotions in self.dataloader_dict['Train']:
            self.Train_Step(
                features= features,
                emotions= emotions
                )

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.gpu_id == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)                
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            'Train.Loss/{}'.format(key.replace('Loss/', '').replace('/', '.')): value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)


            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(self, features, emotions):
        loss_dict = {}
        features = features.to(self.device, non_blocking=True)
        emotions = emotions.to(self.device, non_blocking=True)

        predictions = self.model(features)
        loss_dict['Classification'] = self.criterion(predictions, emotions)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

        return predictions

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, (features, emotions) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            predictions = self.Evaluation_Step(
                features= features,
                emotions= emotions
                )

        if self.gpu_id == 0:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            self.writer_dict['Evaluation'].add_histogram_model(self.model, 'Light_SERNet', self.steps)
        
            distributions = np.zeros(shape= (self.hp.Emotions, self.hp.Emotions))
            for x, y in zip(predictions.argmax(dim= 1), emotions):
                distributions[x, y] += 1

            image_dict = {
                'Distribution': (distributions, None, 'auto', None)
                }
            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        'Evaluation.Loss/{}'.format(key.replace('Loss/', '').replace('/', '.')): value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {
                        'Evaluation.Distribution': wandb.Image(
                            distributions,
                            caption= 'Real'
                            )
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)
        
        self.model.train()


    @torch.no_grad()
    def Inference_Step(self, features, paths):
        features = features.to(self.device, non_blocking=True)

        predictions = self.model(features)
        emotion_info_dict = yaml.load(open(self.hp.Emotion_Info_Path), Loader=yaml.Loader)
        emotion_info_dict = {index: emotion for emotion, index in emotion_info_dict.items()}

        exports = [
            '{}\t{}'.format(path, emotion_info_dict[index])
            for path, index in zip(paths, predictions.argmax(dim= 1).cpu().numpy())
            ]

        os.makedirs(self.hp.Inference_Path, exist_ok= True)
        with open(os.path.join(self.hp.Inference_Path, 'S_{:08d}.txt'.format(self.steps)).replace('\\', '/'), 'a') as f:
            f.write('\n'.join(exports))

    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (fatures, paths) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(
                features= fatures,
                paths= paths
                )

        self.model.train()

    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_dict = torch.load(path, map_location= 'cpu')
        self.model.load_state_dict(state_dict['Model'])
        self.optimizer.load_state_dict(state_dict['Optimizer'])
        self.scheduler.load_state_dict(state_dict['Scheduler'])
        self.steps = state_dict['Steps']

        logging.info('Checkpoint loaded at {} steps in GPU {}.'.format(self.steps, self.gpu_id))

    def Save_Checkpoint(self):
        if self.gpu_id != 0:
            return

        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        state_dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps
            }
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        torch.save(state_dict, checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)


    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            yaml.dump(self.hp, open(hp_path, 'w'))

        if self.steps == 0:
            self.Evaluation_Epoch()
        
        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)

        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    argParser.add_argument('-s', '--steps', default= 0, type= int)    
    argParser.add_argument('-p', '--port', default= 54321, type= int)
    argParser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = argParser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl',
            dist_url= 'tcp://127.0.0.1:{}'.format(args.port)
            )
    new_Trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    new_Trainer.Train()