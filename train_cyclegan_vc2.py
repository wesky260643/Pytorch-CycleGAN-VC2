import librosa
import os
import numpy as np
import argparse
import torch
import time
import pickle

import preprocess
from trainingDataset import trainingDataset
from cyclegan_vc2 import Generator, Discriminator
from logger import Logger

import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed

import horovod.torch as hvd

import functools
print = functools.partial(print, flush=True)


hvd.init()

class CycleGANTraining:
    def __init__(self, args):
        self.start_epoch = 0
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.dataset_A = self.loadPickleFile(args.coded_sps_A_norm)
        self.dataset_B = self.loadPickleFile(args.coded_sps_B_norm)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Speech Parameters
        logf0s_normalization = np.load(args.logf0s_normalization)
        self.log_f0s_mean_A = logf0s_normalization['mean_A']
        self.log_f0s_std_A = logf0s_normalization['std_A']
        self.log_f0s_mean_B = logf0s_normalization['mean_B']
        self.log_f0s_std_B = logf0s_normalization['std_B']

        mcep_normalization = np.load(args.mcep_normalization)
        self.coded_sps_A_mean = mcep_normalization['mean_A']
        self.coded_sps_A_std = mcep_normalization['std_A']
        self.coded_sps_B_mean = mcep_normalization['mean_B']
        self.coded_sps_B_std = mcep_normalization['std_B']

        # Generator and Discriminator
        self.generator_A2B = Generator()
        self.generator_B2A = Generator()
        # self.generator_B2A = self.generator_A2B 
        self.discriminator_A = Discriminator()
        self.discriminator_B = Discriminator()
        # self.discriminator_B = self.discriminator_A 

        # Loss Functions
        criterion_mse = torch.nn.MSELoss()

        # Optimizer
        self.g_params = list(self.generator_A2B.parameters()) + \
                list(self.generator_B2A.parameters())
        self.d_params = list(self.discriminator_A.parameters()) + \
                list(self.discriminator_B.parameters())
        self.g_named_params = list()
        for nparam in self.generator_A2B.named_parameters(prefix="g_a2b"):
            self.g_named_params.append(nparam)
        for nparam in self.generator_B2A.named_parameters(prefix="g_b2a"):
            self.g_named_params.append(nparam)
        self.d_named_params = list()
        for nparam in self.discriminator_A.named_parameters(prefix="d_a"):
            self.d_named_params.append(nparam)
        for nparam in self.discriminator_B.named_parameters(prefix="d_b"):
            self.d_named_params.append(nparam)

        # Initial learning rates
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr

        # Learning rate decay
        self.generator_lr_decay = self.generator_lr / 200000
        self.discriminator_lr_decay = self.discriminator_lr / 200000

        # Starts learning rate decay from after this many iterations have passed
        self.start_decay = args.start_decay

        self.generator_optimizer = torch.optim.Adam(
            self.g_params, lr=self.generator_lr, betas=(args.beta1, args.beta2))
        self.discriminator_optimizer = torch.optim.Adam(
            self.d_params, lr=self.discriminator_lr, betas=(args.beta1, args.beta2))

        # To Load save previously saved models
        self.modelCheckpoint = args.model_checkpoint

        # Validation set Parameters
        self.validation_A_dir = args.validation_A_dir
        self.output_A_dir = args.output_A_dir
        self.validation_B_dir = args.validation_B_dir
        self.output_B_dir = args.output_B_dir

        # Storing Discriminatior and Generator Loss
        self.generator_loss_store = []
        self.discriminator_loss_store = []

        self.file_name = 'log_store_non_sigmoid.txt'
        self.log_dir = args.log_dir
        self.logger = Logger(self.log_dir)

        if args.resume_training_at is not None:
            # Training will resume from previous checkpoint
            self.start_epoch = self.loadModel(args.resume_training_at)
            print("Training resumed")

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def train(self):
        # Training Begins
        torch.manual_seed(args.seed)
        dataset = trainingDataset(datasetA=self.dataset_A,
                                  datasetB=self.dataset_B,
                                  n_frames=128)
        if args.distributed:
            print("------------------- horovod traing setting ----------------")
            print("count of gpu available:", torch.cuda.device_count())
            torch.cuda.set_device(hvd.local_rank())

            self.generator_A2B = self.generator_A2B.cuda()
            self.generator_B2A = self.generator_B2A.cuda()
            self.discriminator_A = self.discriminator_A.cuda()
            self.discriminator_B = self.discriminator_B.cuda()

            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())

            self.generator_optimizer = hvd.DistributedOptimizer(self.generator_optimizer, 
                    named_parameters=self.g_named_params, 
                    compression=hvd.Compression.fp16)
            self.discriminator_optimizer = hvd.DistributedOptimizer(self.discriminator_optimizer, 
                    named_parameters=self.d_named_params, 
                    compression=hvd.Compression.fp16)

            hvd.broadcast_parameters(self.generator_A2B.state_dict(), root_rank=0)
            hvd.broadcast_parameters(self.generator_B2A.state_dict(), root_rank=0)
            hvd.broadcast_parameters(self.discriminator_A.state_dict(), root_rank=0)
            hvd.broadcast_parameters(self.discriminator_B.state_dict(), root_rank=0)

            hvd.broadcast_optimizer_state(self.generator_optimizer, root_rank=0)
            hvd.broadcast_optimizer_state(self.discriminator_optimizer, root_rank=0)
        else:
            self.generator_A2B = self.generator_A2B.to(self.device)
            self.generator_B2A = self.generator_B2A.to(self.device)
            self.discriminator_A = self.discriminator_A.to(self.device)
            self.discriminator_B = self.discriminator_B.to(self.device)

            train_sampler = None
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   drop_last=False,
                                                   pin_memory=True, 
                                                   sampler=train_sampler)
        for epoch in range(self.start_epoch, self.num_epochs):
            self.generator_A2B.train()
            self.generator_B2A.train()
            self.discriminator_A.train()
            self.discriminator_B.train()

            train_sampler.set_epoch(epoch)
            start_time_epoch = time.time()
            # Constants
            cycle_loss_lambda = args.cycle_loss_lambda
            identity_loss_lambda = args.identity_loss_lambda

            # Preparing Dataset
            n_samples = len(self.dataset_A)
            
            for i, (real_A, real_B) in enumerate(train_loader):
                # print("--------------- data size --------------", real_A.size(), real_B.size())
                num_iterations = (
                    n_samples // self.batch_size) * epoch + i
                # print("iteration no: ", num_iterations, epoch)
                if num_iterations > 10000:
                    identity_loss_lambda = 0
                if num_iterations > self.start_decay:
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='discriminator')

                # real_A = real_A.to(self.device).float()
                # real_B = real_B.to(self.device).float()
                real_A = real_A.cuda().float()
                real_B = real_B.cuda().float()

                # Generator Loss function

                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)

                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)

                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)

                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)

                # dot_d_fake_A = self.discriminator_A(cycle_A)
                # dot_d_fake_B = self.discriminator_B(cycle_B)

                # Generator Cycle loss
                cycleLoss = torch.mean(torch.abs(real_A - cycle_A)) + \
                        torch.mean(torch.abs(real_B - cycle_B))

                # Generator Identity Loss
                identiyLoss = torch.mean(torch.abs(real_A - identity_A)) + \
                        torch.mean(torch.abs(real_B - identity_B))

                # Generator Loss
                generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                # two-step generator loss
                # two_step_generator_loss_A = torch.mean((1 - dot_d_fake_A) ** 2)
                # two_step_generator_loss_B = torch.mean((1 - dot_d_fake_B) ** 2)

                # Total Generator Loss
                generator_loss = generator_loss_A2B + generator_loss_B2A + \
                    cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
                        # two_step_generator_loss_A + two_step_generator_loss_B + \
                self.generator_loss_store.append(generator_loss.item())

                # Backprop for Generator
                # self.reset_grad()
                self.generator_optimizer.synchronize()
                self.generator_optimizer.zero_grad()
                generator_loss.backward()
                # if num_iterations > self.start_decay:  # Linearly decay learning rate
                #     self.adjust_lr_rate(
                #         self.generator_optimizer, name='generator')

                self.generator_optimizer.step()
                # tensorboard log
                loss = dict()
                loss["G/cycleLoss"] = cycleLoss.item()
                loss["G/identiyLoss"] = identiyLoss.item()
                loss["G/generator_loss"] = generator_loss.item()
                loss["G/generator_loss_A2B"] = generator_loss_A2B.item()
                loss["G/generator_loss_B2A"] = generator_loss_B2A.item()
                # loss["G/two_step_generator_loss_A"] = two_step_generator_loss_A.item()
                # loss["G/two_step_generator_loss_B"] = two_step_generator_loss_B.item()

                # Discriminator Loss Function

                # Discriminator Feed Forward
                d_real_A = self.discriminator_A(real_A)
                generated_A = self.generator_B2A(real_B)
                d_fake_A = self.discriminator_A(generated_A)

                d_real_B = self.discriminator_B(real_B)
                generated_B = self.generator_A2B(real_A)
                d_fake_B = self.discriminator_B(generated_B)
                
                # Loss Functions
                d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                # two step loss function
                # dot_d_A_real = self.discriminator_A(real_A)
                # dot_d_fake_B = self.generator_A2B(real_A)
                # dot_d_cycle_A = self.generator_B2A(dot_d_fake_B)
                # dot_d_A_fake = self.discriminator_A(dot_d_cycle_A)

                # dot_d_B_real = self.discriminator_B(real_B)
                # dot_d_fake_A = self.generator_B2A(real_B)
                # dot_d_cycle_B = self.generator_A2B(dot_d_fake_A)
                # dot_d_B_fake = self.discriminator_B(dot_d_cycle_B)

                # two_step_d_loss_A_real = torch.mean((1 - dot_d_A_real) ** 2)
                # two_step_d_loss_A_fake = torch.mean((0 - dot_d_A_fake) ** 2)
                # two_step_d_loss_A = (two_step_d_loss_A_real + two_step_d_loss_A_fake) / 2
                # 
                # two_step_d_loss_B_real = torch.mean((1 - dot_d_B_real) ** 2)
                # two_step_d_loss_B_fake = torch.mean((0 - dot_d_B_fake) ** 2)
                # two_step_d_loss_B = (two_step_d_loss_B_real + two_step_d_loss_B_fake) / 2

                # Final Loss for discriminator
                # d_loss = (d_loss_A + d_loss_B) / 2.0 + (two_step_d_loss_A + two_step_d_loss_B) / 2
                d_loss = (d_loss_A + d_loss_B) / 2.0
                self.discriminator_loss_store.append(d_loss.item())

                # Backprop for Discriminator
                # self.reset_grad()
                self.discriminator_optimizer.synchronize()
                self.discriminator_optimizer.zero_grad()
                d_loss.backward()

                # if num_iterations > self.start_decay:  # Linearly decay learning rate
                #     self.adjust_lr_rate(
                #         self.discriminator_optimizer, name='discriminator')

                self.discriminator_optimizer.step()
                # tensorboard log 
                loss["D/d_loss_A_real"] = d_loss_A_real.item()
                loss["D/d_loss_A_fake"] = d_loss_A_fake.item()
                loss["D/d_loss_A"] = d_loss_A.item()
                loss["D/d_loss_B_real"] = d_loss_B_real.item()
                loss["D/d_loss_B_fake"] = d_loss_B_fake.item()
                loss["D/d_loss_B"] = d_loss_B.item()
                loss["D/d_loss"] = d_loss.item()
                # loss["D/two_step_d_loss_A_real"] = two_step_d_loss_A_real.item()
                # loss["D/two_step_d_loss_A_fake"] = two_step_d_loss_A_fake.item()
                # loss["D/two_step_d_loss_A"] = two_step_d_loss_A.item()
                # loss["D/two_step_d_loss_B_real"] = two_step_d_loss_B_real.item()
                # loss["D/two_step_d_loss_B_fake"] = two_step_d_loss_B_fake.item()
                # loss["D/two_step_d_loss_B"] = two_step_d_loss_B.item()

                if num_iterations % 50 == 0:
                    store_to_file = "Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                        num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B, generator_loss_B2A, identiyLoss, cycleLoss, d_loss_A, d_loss_B)
                    print("Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} \tGA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                        num_iterations, generator_loss.item(), d_loss.item(), generator_loss_A2B, generator_loss_B2A, identiyLoss, cycleLoss, d_loss_A, d_loss_B))
                    self.store_to_file(store_to_file)
                if num_iterations % 10 == 0:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, num_iterations + 1)
            end_time = time.time()
            store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n".format(
                epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch)
            self.store_to_file(store_to_file)
            print("Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n".format(
                epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch))

            if epoch % 100 == 0 and epoch != 0:
                # Save the Entire model
                print("Saving model Checkpoint  ......")
                store_to_file = "Saving model Checkpoint  ......"
                self.store_to_file(store_to_file)
                self.saveModelCheckPoint(epoch, '{}'.format(self.modelCheckpoint + '_CycleGAN_CheckPoint.ep' + str(epoch)))
                print("Model Saved!")

            if epoch % 100 == 0 and epoch != 0:
                # Validation Set
                validation_start_time = time.time()
                self.validation_for_A_dir()
                self.validation_for_B_dir()
                validation_end_time = time.time()
                store_to_file = "Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time)
                self.store_to_file(store_to_file)
                print("Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time))

    def validation_for_A_dir(self):
        num_mcep = 36
        sampling_rate = 22000
        frame_period = 5.0
        n_frames = 128
        validation_A_dir = self.validation_A_dir
        output_A_dir = self.output_A_dir

        print("Generating Validation Data B from A...")
        for idx, file in enumerate(os.listdir(validation_A_dir)):
            if idx > 10: break
            filePath = os.path.join(validation_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_A,
                                                       std_log_src=self.log_f0s_std_A,
                                                       mean_log_target=self.log_f0s_mean_B,
                                                       std_log_target=self.log_f0s_std_B)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            try:
                librosa.output.write_wav(path=os.path.join(output_A_dir, os.path.basename(file)),
                                         y=wav_transformed,
                                         sr=sampling_rate)
            except:
                print("write wave file error")

    def validation_for_B_dir(self):
        num_mcep = 36
        sampling_rate = 22000
        frame_period = 5.0
        n_frames = 128
        validation_B_dir = self.validation_B_dir
        output_B_dir = self.output_B_dir

        print("Generating Validation Data A from B...")
        for idx, file in enumerate(os.listdir(validation_B_dir)):
            if idx > 10 : break
            filePath = os.path.join(validation_B_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_B,
                                                       std_log_src=self.log_f0s_std_B,
                                                       mean_log_target=self.log_f0s_mean_A,
                                                       std_log_target=self.log_f0s_std_A)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_B_mean) / self.coded_sps_B_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_B2A(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.coded_sps_A_std + self.coded_sps_A_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            try:
                librosa.output.write_wav(path=os.path.join(output_B_dir, os.path.basename(file)),
                                         y=wav_transformed,
                                         sr=sampling_rate)
            except:
                print("write wave file error")

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'discriminator_loss_store': self.discriminator_loss_store,
            'model_genA2B_state_dict': self.generator_A2B.state_dict(),
            'model_genB2A_state_dict': self.generator_B2A.state_dict(),
            'model_discriminatorA': self.discriminator_A.state_dict(),
            'model_discriminatorB': self.discriminator_B.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A.load_state_dict(
            state_dict=checkPoint['model_discriminatorA'])
        self.discriminator_B.load_state_dict(
            state_dict=checkPoint['model_discriminatorB'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']
        return epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train CycleGAN using source dataset and target dataset")

    logf0s_normalization_default = 'data/cache_check/logf0s_normalization.npz'
    mcep_normalization_default = 'data/cache_check/mcep_normalization.npz'
    coded_sps_A_norm = 'data/cache_check/coded_sps_A_norm.pickle'
    coded_sps_B_norm = 'data/cache_check/coded_sps_B_norm.pickle'
    model_checkpoint = 'data/cache_check/model_checkpoint/'
    # resume_training_at = 'data/cache_check/model_checkpoint/_CycleGAN_CheckPoint'

    validation_A_dir_default = 'data/vcc2018_training.speakers/VCC2SF1/'
    output_A_dir_default = 'data/vcc2018_training.speakers/converted_sound/VCC2SF1'

    validation_B_dir_default = 'data/vcc2018_training.speakers/VCC2TM2/'
    output_B_dir_default = 'data/vcc2018_training.speakers/converted_sound/VCC2TM2/'
    # train dir args
    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--coded_sps_A_norm', type=str,
                        help="mcep norm for data A", default=coded_sps_A_norm)
    parser.add_argument('--coded_sps_B_norm', type=str,
                        help="mcep norm for data B", default=coded_sps_B_norm)
    parser.add_argument('--model_checkpoint', type=str,
                        help="location where you want to save the model", default=model_checkpoint)
    parser.add_argument('--resume_training_at', type=str,
                        help="Location of the pre-trained model to resume training", default=None)
    parser.add_argument('--validation_A_dir', type=str,
                        help="validation set for sound source A", default=validation_A_dir_default)
    parser.add_argument('--output_A_dir', type=str,
                        help="output for converted Sound Source A", default=output_A_dir_default)
    parser.add_argument('--validation_B_dir', type=str,
                        help="Validation set for sound source B", default=validation_B_dir_default)
    parser.add_argument('--output_B_dir', type=str,
                        help="Output for converted sound Source B", default=output_B_dir_default)
    parser.add_argument('--log_dir', type=str,
                        help="tensorboard log dir", default="./logs")

    # model config
    parser.add_argument("--cycle_loss_lambda", default=10, type=float, help="weight for cycle loss ")
    parser.add_argument("--identity_loss_lambda", default=5, type=float, help="weight for identity loss ")

    # train config
    parser.add_argument("--num_epochs", default=5000, type=int, help="num of total epochs to train")
    parser.add_argument("--batch_size", default=8, type=int, help="mini batch size")
    parser.add_argument("--generator_lr", default=0.0002, type=float, help="learning rate for Generator")
    parser.add_argument("--discriminator_lr", default=0.0001, type=float, help="learning rate for Discriminator")
    parser.add_argument("--start_decay", default=12500, type=int, help="num of iterations to decay G_lr and D_lr")
    parser.add_argument("--beta1", default=0.5, type=float, help="beta1 for Adam optimizer")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2 for Adam optimizer")
    parser.add_argument("--seed", default=20, type=int, help="random seed")
    
    # distribute data parallel args
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://103.97.83.4:29503', type=str,
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:30003', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_false',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("-" * 30 + "train args" + "-"*30)
    print(args)
    print("-" * 30 + "train args" + "-"*30)
    
    if not os.path.exists(args.output_A_dir):
        os.makedirs(args.output_A_dir)
    if not os.path.exists(args.output_B_dir):
        os.makedirs(args.output_B_dir)
    if not os.path.exists(args.model_checkpoint):
        os.makedirs(args.model_checkpoint)

    # Check whether following cached files exists
    if not os.path.exists(args.logf0s_normalization) or not os.path.exists(args.mcep_normalization):
        print( "Cached files do not exist, please run the program preprocess_training.py first" )

    cycleGAN = CycleGANTraining(args)
    cycleGAN.train()
    




