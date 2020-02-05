import random
import math
import os
from argparse import ArgumentParser

# Torch
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# TensorboardX
from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from vae.models import VAE
from vae.datasets import DebugDatset, FoldersDataset
from vae import utils

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

# Random seeding
random.seed(99)
torch.manual_seed(99)
if cuda: torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = ArgumentParser(description='VAE training')
    parser.add_argument('--run_name', required=True, type=str,
                        help='name of the current run (where runs are saved)')
    parser.add_argument('--data_dir', required=True, type=str, help='directory of the data')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs run')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--resolution', type=int, default=64, help='resolution of input images')
    parser.add_argument('--r_dim', type=int, default=256, help='r_dim for VAE')
    parser.add_argument('--h_dim', type=int, default=128, help='h_dim for VAE')
    parser.add_argument('--z_dim', type=int, default=3, help='z_dim for VAE')
    parser.add_argument('--l', type=int, default=8, help='L for VAE')
    parser.add_argument('--data_parallel', type=bool, default=False, help='whether to parallelise based on data')
    args = parser.parse_args()

    # Make the run directory
    save_dir = os.path.join('vae/saved_runs', args.run_name)
    os.mkdir(save_dir)

    # Load the dataset
    # train_dataset = DebugDatset(data_dir=args.data_dir, resolution=args.resolution)
    # val_dataset = DebugDatset(data_dir=args.data_dir, resolution=args.resolution)
    train_dataset = FoldersDataset(os.path.join(args.data_dir, 'train'), resolution=args.resolution)
    val_dataset = FoldersDataset(os.path.join(args.data_dir, 'val'), resolution=args.resolution)

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    # Create model and optimizer
    model = VAE(c_dim=3, r_dim=args.r_dim, h_dim=args.h_dim, z_dim=args.z_dim, l=args.l).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = utils.AnnealingStepLR(optimizer, mu_i=args.lr, mu_f=args.lr/10, n=1.6e6)
    sigma_scheduler = utils.AnnealingStepSigma(2.0, 0.7, 2e5)


    def step(engine, batch):
        model.train()

        x = batch
        x = x.to(device)

        # Reconstruction, representation and divergence
        x_mu, _, kl = model(x)

        # Log likelihood
        sigma = sigma_scheduler.sigma
        lr = lr_scheduler.get_lr()[0]
        ll = Normal(x_mu, sigma).log_prob(x)

        likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        # Evidence lower bound
        elbo = likelihood - kl_divergence
        loss = -elbo
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        lr_scheduler.step()
        sigma_scheduler.step()

        return {'elbo': elbo.item(), 'likelihood': likelihood.item(), 'kl': kl_divergence.item(),
                'lr': lr, 'sigma': sigma}


    # Trainer and metrics
    trainer = Engine(step)
    metric_names = ['elbo', 'likelihood', 'kl', 'lr', 'sigma']
    RunningAverage(output_transform=lambda x: x['elbo']).attach(trainer, 'elbo')
    RunningAverage(output_transform=lambda x: x['likelihood']).attach(trainer, 'likelihood')
    RunningAverage(output_transform=lambda x: x['kl']).attach(trainer, 'kl')
    RunningAverage(output_transform=lambda x: x['lr']).attach(trainer, 'lr')
    RunningAverage(output_transform=lambda x: x['sigma']).attach(trainer, 'sigma')
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Model checkpointing
    checkpoint_handler = ModelCheckpoint(os.path.join(save_dir, 'checkpoints'), 'vae',
                                         save_interval=1, n_saved=3, require_empty=False)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'model': model, 'optimizer': optimizer,
                                       'lr_scheduler': lr_scheduler, 'sigma_scheduler': sigma_scheduler})

    timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                                       pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Tensorbard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        if engine.state.iteration % 100 == 0:
            for metric, value in engine.state.metrics.items():
                writer.add_scalar('training/{}'.format(metric), value, engine.state.iteration)


    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.eval()
        with torch.no_grad():
            x = next(iter(val_loader))
            x = x.to(device)

            # Reconstruction, representation and divergence
            x_mu, _, kl = model(x)

            # Validate at last sigma
            ll = Normal(x_mu, sigma_scheduler.sigma).log_prob(x)

            likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
            kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

            # Evidence lower bound
            elbo = likelihood - kl_divergence

            writer.add_scalar('validation/elbo', elbo.item(), engine.state.epoch)
            writer.add_scalar('validation/likelihood', likelihood.item(), engine.state.epoch)
            writer.add_scalar('validation/kl', kl_divergence.item(), engine.state.epoch)

            save_images(engine, x)


    def save_images(engine, x):
        x_mu, r = model.sample(x)

        r = r.view(-1, 1, int(math.sqrt(args.r_dim)), int(math.sqrt(args.r_dim)))

        x_mu = x_mu.detach().cpu().float()
        r = r.detach().cpu().float()

        writer.add_image('representation', make_grid(r), engine.state.epoch)
        writer.add_image('generation', make_grid(x_mu), engine.state.epoch)
        writer.add_image('query', make_grid(x), engine.state.epoch)


    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, {'model_exception': model})
        else:
            raise e


    trainer.run(train_loader, args.n_epochs)
    writer.close()
