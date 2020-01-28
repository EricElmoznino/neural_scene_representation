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

from gqn import GenerativeQueryNetwork
from gqn.datasets import DebugDatset, XYRHDataset, partition
from gqn import utils

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')

# Random seeding
random.seed(99)
torch.manual_seed(99)
if cuda: torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = ArgumentParser(description='Generative Query Network training')
    parser.add_argument('--run_name', required=True, type=str,
                        help='name of the current run (where runs are saved)')
    parser.add_argument('--data_dir', required=True, type=str, help='directory of the data')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs run')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--resolution', type=int, default=64, help='resolution of input images')
    parser.add_argument('--max_viewpoints', type=int, default=7,
                        help='maximum number of viewpoints for a given scene')
    parser.add_argument('--min_viewpoints', type=int, default=3,
                        help='minimum number of viewpoints for a given scene')
    parser.add_argument('--r_dim', type=int, default=256, help='r_dim for GQN')
    parser.add_argument('--h_dim', type=int, default=128, help='h_dim for GQN')
    parser.add_argument('--z_dim', type=int, default=64, help='z_dim for GQN')
    parser.add_argument('--l', type=int, default=8, help='L for GQN')
    parser.add_argument('--data_parallel', type=bool, default=False, help='whether to parallelise based on data')
    args = parser.parse_args()

    # Make the run directory
    save_dir = os.path.join('gqn/saved_runs', args.run_name)
    os.mkdir(save_dir)

    # Load the dataset
    # train_dataset = DebugDatset(data_dir=args.data_dir, resolution=args.resolution, max_viewpoints=args.max_viewpoints)
    # val_dataset = DebugDatset(data_dir=args.data_dir, resolution=args.resolution, max_viewpoints=args.max_viewpoints)
    train_dataset = XYRHDataset(os.path.join(args.data_dir, 'train'), resolution=args.resolution,
                                max_viewpoints=args.max_viewpoints)
    val_dataset = XYRHDataset(os.path.join(args.data_dir, 'val'), resolution=args.resolution,
                              max_viewpoints=args.max_viewpoints)

    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    # Create model and optimizer
    model = GenerativeQueryNetwork(c_dim=3, v_dim=train_dataset.v_dim,
                                   r_dim=args.r_dim, h_dim=args.h_dim, z_dim=args.z_dim, l=args.l).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Rate annealing schemes
    sigma_scheme = utils.Annealer(2.0, 0.7, 80000)


    def step(engine, batch):
        model.train()

        x, v = batch
        x, v = x.to(device), v.to(device)
        x, v, x_q, v_q = partition(x, v, args.min_viewpoints)

        # Reconstruction, representation and divergence
        x_mu, _, kl = model(x, v, x_q, v_q)

        # Log likelihood
        sigma = next(sigma_scheme)
        ll = Normal(x_mu, sigma).log_prob(x_q)

        likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        # Evidence lower bound
        elbo = likelihood - kl_divergence
        loss = -elbo
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return {'elbo': elbo.item(), 'likelihood': likelihood.item(), 'kl': kl_divergence.item(), 'sigma': sigma}


    # Trainer and metrics
    trainer = Engine(step)
    metric_names = ['elbo', 'likelihood', 'kl', 'sigma']
    RunningAverage(output_transform=lambda x: x['elbo']).attach(trainer, 'elbo')
    RunningAverage(output_transform=lambda x: x['likelihood']).attach(trainer, 'likelihood')
    RunningAverage(output_transform=lambda x: x['kl']).attach(trainer, 'kl')
    RunningAverage(output_transform=lambda x: x['sigma']).attach(trainer, 'sigma')
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Model checkpointing
    checkpoint_handler = ModelCheckpoint(os.path.join(save_dir, 'checkpoints'), 'gqn',
                                         save_interval=1, n_saved=3, require_empty=False)
    trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                              to_save={'model': model, 'optimizer': optimizer, 'sigma_annealer': sigma_scheme})

    timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                                       pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Tensorbard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        for metric, value in engine.state.metrics.items():
            writer.add_scalar('training/{}'.format(metric), value, engine.state.iteration)


    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        model.eval()
        with torch.no_grad():
            x, v = next(iter(val_loader))
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v, args.min_viewpoints)

            # Reconstruction, representation and divergence
            x_mu, _, kl = model(x, v, x_q, v_q)

            # Validate at last sigma
            ll = Normal(x_mu, sigma_scheme.recent).log_prob(x_q)

            likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
            kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

            # Evidence lower bound
            elbo = likelihood - kl_divergence

            writer.add_scalar('validation/elbo', elbo.item(), engine.state.epoch)
            writer.add_scalar('validation/likelihood', likelihood.item(), engine.state.epoch)
            writer.add_scalar('validation/kl', kl_divergence.item(), engine.state.epoch)


    @trainer.on(Events.EPOCH_COMPLETED)
    def save_images(engine):
        model.eval()
        with torch.no_grad():
            x, v = engine.state.batch
            x, v = x.to(device), v.to(device)
            x, v, x_q, v_q = partition(x, v, args.min_viewpoints)

            x_mu, r = model.sample(x, v, v_q)

            r = r.view(-1, 1, int(math.sqrt(args.r_dim)), int(math.sqrt(args.r_dim)))

            x_mu = x_mu.cpu().float()
            r = r.cpu().float()

            writer.add_image('representation', make_grid(r), engine.state.epoch)
            writer.add_image('reconstruction', make_grid(x_mu), engine.state.epoch)
            writer.add_image('query', make_grid(x_q), engine.state.epoch)


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
