import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average #  , Accuracy
from vgn.utils.accuracy import Accuracy
import torch
from torch.utils import tensorboard
import torch.nn.functional as F

from vgn.dataset import Dataset
from vgn.networks import get_network
from vgn.detection import predict, process, select as select_grasps


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}
    torch.manual_seed(args.seed)

    # create log directory
    time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
    description = "{}_dataset={},augment={},net={},batch_size={},lr={:.0e},{}".format(
        time_stamp,
        args.dataset.name,
        args.augment,
        args.net,
        args.batch_size,
        args.lr,
        args.description,
    ).strip(",")
    logdir = args.logdir / description

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.batch_size, args.val_split, args.augment, kwargs
    )

    # build the network
    net = get_network(args.net, **{'num_qual_heads': args.num_qual_heads}).to(device)
    ensemble = (args.net == "ensembleconv")

    # define optimizer and metrics
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    diff = args.batch_size//args.num_qual_heads
    if ensemble:
        metrics_trainer = {
            "loss": Average(lambda out: out[3], device=device),
            "accuracy1": Accuracy(lambda out: (torch.round(out[1][0][:diff]) > 0.5, out[2][0][:diff] > 0.5), device=device),
            "accuracy2": Accuracy(lambda out: (torch.round(out[1][0][diff:diff*2]) > 0.5, out[2][0][diff:diff*2] > 0.5), device=device),
            "accuracy3": Accuracy(lambda out: (torch.round(out[1][0][diff*2:diff*3]) > 0.5, out[2][0][diff*2:diff*3] > 0.5), device=device),
            "accuracy4": Accuracy(lambda out: (torch.round(out[1][0][diff*3:diff*4]) > 0.5, out[2][0][diff*3:diff*4] > 0.5), device=device),
            # "accuracy5": Accuracy(lambda out: (torch.round(out[1][0][:,4]) > 0.5, out[2][0] > 0.5), device=device),
        }
        metrics_evaluator = {
            "loss": Average(lambda out: out[3], device=device),
            "accuracy1": Accuracy(lambda out: (torch.round(out[1][0][:,0]) > 0.5, out[2][0] > 0.5), device=device),
            "accuracy2": Accuracy(lambda out: (torch.round(out[1][0][:,1]) > 0.5, out[2][0] > 0.5), device=device),
            "accuracy3": Accuracy(lambda out: (torch.round(out[1][0][:,2]) > 0.5, out[2][0] > 0.5), device=device),
            "accuracy4": Accuracy(lambda out: (torch.round(out[1][0][:,3]) > 0.5, out[2][0] > 0.5), device=device),
        }
        # for i in range(args.num_qual_heads):
        #     metrics[f"accuracy{i}"] = Accuracy(lambda out: (torch.round(out[1][0][:,i]).int(), out[2][0].int()), device=device)
    else:
        metrics = {
            "loss": Average(lambda out: out[3], device=device),
            "accuracy": Accuracy(lambda out: (torch.round(out[1][0]) > 0.5, out[2][0] > 0.5), device=device),
        }
        metrics_trainer = metrics
        metrics_evaluator = metrics

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, loss_fn, metrics_trainer, device, ensemble)
    evaluator = create_evaluator(net, loss_fn, metrics_evaluator, device, ensemble)

    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True).attach(trainer)

    train_writer, val_writer = create_summary_writers(net, device, logdir)

    if ensemble:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_train_results(engine):
            epoch, metrics = trainer.state.epoch, trainer.state.metrics
            train_writer.add_scalar("loss", metrics["loss"], epoch)
            for i in range(args.num_qual_heads):
                train_writer.add_scalar(f"accuracy{i+1}", metrics[f"accuracy{i+1}"], epoch)
            # train_writer.add_scalar("accuracy1", metrics["accuracy1"], epoch)
            # train_writer.add_scalar("accuracy2", metrics["accuracy2"], epoch)
            # train_writer.add_scalar("accuracy3", metrics["accuracy3"], epoch)
            # train_writer.add_scalar("accuracy4", metrics["accuracy4"], epoch)
            # train_writer.add_scalar("accuracy5", metrics["accuracy5"], epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            epoch, metrics = trainer.state.epoch, evaluator.state.metrics
            val_writer.add_scalar("loss", metrics["loss"], epoch)
            for i in range(args.num_qual_heads):
                val_writer.add_scalar(f"accuracy{i+1}", metrics[f"accuracy{i+1}"], epoch)
            # add the grasp locations and scores to the tensorboard
            # val_writer.add_image(
            #     "grasp_locations",
            #     make_grasp_locations_image(
            #         net, val_loader, device, args.num_qual_heads, args.num_grasps
            #     ),
            #     epoch,
            # )
            # val_writer.add_scalar("accuracy1", metrics["accuracy1"], epoch)
            # val_writer.add_scalar("accuracy2", metrics["accuracy2"], epoch)
            # val_writer.add_scalar("accuracy3", metrics["accuracy3"], epoch)
            # val_writer.add_scalar("accuracy4", metrics["accuracy4"], epoch)
            # val_writer.add_scalar("accuracy5", metrics["accuracy5"], epoch)
        
        @evaluator.on(Events.ITERATION_COMPLETED)
        def log_grasps_qual(engine):
            # add the grasp locations and scores to the tensorboard as a 3d scatter plot
            val_writer.add_figure(
                "grasp_locations_qual", 
                make_grasp_locations_qual_figure(
                    net, engine.state.output, device, args.num_qual_heads
                ), 
                engine.state.epoch*engine.state.iteration,
            )

    else:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_train_results(engine):
            epoch, metrics = trainer.state.epoch, trainer.state.metrics
            train_writer.add_scalar("loss", metrics["loss"], epoch)
            train_writer.add_scalar("accuracy", metrics["accuracy"], epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            epoch, metrics = trainer.state.epoch, evaluator.state.metrics
            val_writer.add_scalar("loss", metrics["loss"], epoch)
            val_writer.add_scalar("accuracy", metrics["accuracy"], epoch)
            
    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "vgn",
        n_saved=100,
        require_empty=True,
        save_as_state_dict=True,
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


# create a figure with the grasp locations and scores
# note that output is (x, y_pred, y, loss)
def make_grasp_locations_qual_figure(net, output, device, num_qual_heads):
    # prepare the batch for the network
    tsdf = output[0]
    tsdf = tsdf.to(device)
    # run the network
    fig = plt.figure(figsize=(10, 10))
    with torch.no_grad():
        qual_vol, rot_vol_orig, width_vol_orig = predict(tsdf, net, device, validate=True)
        for i in range(0, num_qual_heads):
            # for j in range(len(qual_vol)):
            batch_idx = np.random.randint(0, len(qual_vol))
            q_vol = qual_vol[batch_idx,i]
            q_vol, rot_vol, width_vol = process(tsdf.cpu().detach().numpy()[batch_idx], q_vol, rot_vol_orig[batch_idx], width_vol_orig[batch_idx])
            grasps, scores = select_grasps(q_vol, rot_vol, width_vol)
            positions = np.array([grasp.pose.translation for grasp in grasps]) if len(grasps) > 0 else np.zeros((1, 3))
            ax1 = fig.add_subplot(100 + 20*num_qual_heads + i*2 + 1, projection="3d")
            ax1.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=scores,
                cmap=cm.coolwarm
            )
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            ax1.set_xlim3d(0, 0.3)
            ax1.set_ylim3d(0, 0.3)
            ax1.set_zlim3d(0, 0.3)
            ax1.set_title(f"Grasps {i}")
            
            voxel_grid = tsdf[batch_idx,0].cpu().detach().numpy()
            points = voxel_grid.nonzero()
            ax2 = fig.add_subplot(100 + 20*num_qual_heads + i*2 + 2, projection="3d")
            ax2.scatter(points[0], points[1], points[2], c=voxel_grid[points[0], points[1], points[2]], cmap=cm.coolwarm)
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("z")
            ax2.set_xlim3d(0, 40)
            ax2.set_ylim3d(0, 40)
            ax2.set_zlim3d(0, 40)
            ax2.set_title(f"TSDF {i}")
                
    return fig


def create_train_val_loaders(root, batch_size, val_split, augment, kwargs):
    # load the dataset
    dataset = Dataset(root, augment=augment)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    tsdf, (label, rotations, width), index = batch
    tsdf = tsdf.to(device)
    label = label.float().to(device)
    rotations = rotations.to(device)
    width = width.to(device)
    index = index.to(device)
    return tsdf, (label, rotations, width), index


def select(out, index):
    qual_out, rot_out, width_out = out
    batch_index = torch.arange(qual_out.shape[0])
    label = qual_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
    rot = rot_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]]
    width = width_out[batch_index, :, index[:, 0], index[:, 1], index[:, 2]].squeeze()
    return label, rot, width


def loss_fn(y_pred, y, ensemble=False):
    label_pred, rotation_pred, width_pred = y_pred
    label, rotations, width = y
    if not ensemble:
        loss_qual = _qual_loss_fn(label_pred, label)
    else:
        loss_qual = _qual_loss_fn_ensemble(label_pred, label)
    loss_rot = _rot_loss_fn(rotation_pred, rotations)
    loss_width = _width_loss_fn(width_pred, width)
    loss = loss_qual + label * (loss_rot + 0.01 * loss_width)
    return loss.mean()


def _qual_loss_fn(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="none")


def _qual_loss_fn_ensemble(pred, target):
    return torch.sum(torch.cat([F.binary_cross_entropy(pred[:,i], target, reduction="none").unsqueeze(0) for i in range(pred.shape[1])], dim=0), dim=0) / pred.shape[1]


def _rot_loss_fn(pred, target):
    loss0 = _quat_loss_fn(pred, target[:, 0])
    loss1 = _quat_loss_fn(pred, target[:, 1])
    return torch.min(loss0, loss1)


def _quat_loss_fn(pred, target):
    return 1.0 - torch.abs(torch.sum(pred * target, dim=1))


def _width_loss_fn(pred, target):
    return F.mse_loss(pred, target, reduction="none")


def create_trainer(net, optimizer, loss_fn, metrics, device, ensemble=False):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        x, y, index = prepare_batch(batch, device)
        qual_out, rot_out, width_out = net(x)
        # next_head = torch.randint(0, qual_out.shape[1], (1,)).item()
        # y_pred = select(net(x), index)
        diff = qual_out.shape[0]//qual_out.shape[1]
        qual = torch.cat(
            [
                qual_out[:,j][diff*j:diff*(j+1),None,...] for j in range(qual_out.shape[1])
            ],
            dim=0,
        )
        y_pred = select((qual, rot_out, width_out), index)
        loss = loss_fn(y_pred, y) #  , ensemble=ensemble)

        # backward
        loss.backward()
        optimizer.step()

        # qual = []
        # for j in range(qual_out.shape[1]):
        #     if j == next_head:
        #         qual.append(y_pred[0][:,None,...])
        #     else:
        #         qual.append(-torch.ones(y_pred[0].shape, device=device).unsqueeze(1))
        # y_pred = (torch.cat(qual, dim=1), y_pred[1], y_pred[2])
        # y_pred = (torch.cat([y_pred[0].unsqueeze(1), -torch.ones(y_pred[0].shape, device=device).unsqueeze(1)], dim=1), y_pred[1], y_pred[2])

        return x, y_pred, y, loss

    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, loss_fn, metrics, device, ensemble=False):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, index = prepare_batch(batch, device)
            y_pred = select(net(x), index)
            loss = loss_fn(y_pred, y, ensemble=ensemble)
        return x, y_pred, y, loss

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--net", default="conv")
    parser.add_argument("--net", default="ensembleconv")
    parser.add_argument("--num_qual_heads", type=int, default=4)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/runs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    main(args)
