import os
import sys
import signal
import pyrallis
from config import TrainConfig
import torch
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from vit import VisionTransformer
from loss import OHEM_CELoss
from cf_dataloader import CFDataset


from models.resnet import ResNet50


@pyrallis.wrap()
def main(train_cfg:TrainConfig):
    def get_state_dict(model, optimizer, epoch):
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        return state_dict

    def load_state_dict(model, optimizer, state_dict):
        missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            optimizer.load_state_dict(state_dict["optimizer"])
        else:
            # set group lr for model.patchify and model.blocks
            lr = train_cfg.optim.lr
            optimizer = optim.Adam(
                [
                    {"params": model.patchify.parameters(), "lr": lr * 1e-2},
                    {"params": model.blocks.parameters(), "lr": lr * 1e-2},
                    {"params": model.header.parameters(), "lr": lr},
                ],
                lr=1e-3,
                weight_decay=1e-4
            )
        return model, optimizer, state_dict["epoch"]

    def cls_correct_repr(correct):
        str_repr = ""
        for i in range(len(correct)):
            str_repr += f"[{i}: {correct[i]:.5f}], "
        return str_repr


    model = VisionTransformer(
        num_classes=train_cfg.model.num_classes,
        in_channels=train_cfg.model.in_channels,
        img_size=train_cfg.model.img_size,
        patch_size=train_cfg.model.patch_size,
        d_model=train_cfg.model.d_model,
        num_heads=train_cfg.model.num_heads,
        num_layers=train_cfg.model.num_layers,
        ffn_hidden_channels=train_cfg.model.ffn_hidden_channels,
        dropout=train_cfg.model.dropout,
        classifier=train_cfg.model.classifier
    ).to(train_cfg.device)
    # model = ResNet50().to(train_cfg.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg.optim.lr,
        weight_decay=train_cfg.optim.weight_decay
    )

    try:
        ckpt_path = train_cfg.ckpt_dir / "last.pt"
        state_dict = torch.load(ckpt_path)
        model, optimizer, epoch = load_state_dict(model, optimizer, state_dict)
        print(f"Load checkpoint from {ckpt_path}")
    except:
        epoch = 0
        print("No valid checkpoint found, start from scratch")


    train_dataset = CFDataset(
        train_cfg.data.train_data_path,
        train=True,
        load_img2mem=train_cfg.data.load_img2mem,
        augment=train_cfg.data.augment,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        # pin_memory=True,
        # pin_memory_device=train_cfg.device
    )

    val_dataset = CFDataset(
        train_cfg.data.val_data_path,
        train=False,
        load_img2mem=train_cfg.data.load_img2mem,
        augment=train_cfg.data.augment,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        # pin_memory_device=train_cfg.device
    )

    if train_cfg.model.classifier:
        if train_cfg.loss_augment:
            cls_weight = torch.tensor([v for (_, v) in train_dataset.cls_cnt])
            cls_weight = cls_weight.sum() / cls_weight
            cls_weight = cls_weight / cls_weight.sum()
            print("Using Weighted OHEM_CELoss")
            print(f"Class weight: {cls_weight}")
            # ce_loss_fn = torch.nn.CrossEntropyLoss(cls_weight.to(train_cfg.device))
            ce_loss_fn = OHEM_CELoss(ratio=0.5, weight=cls_weight.to(train_cfg.device))
        else:
            print("Using vanilla CrossEntropyLoss")
            ce_loss_fn = torch.nn.CrossEntropyLoss()
        print("Training Classifier")

    else:
        ce_loss_fn = torch.nn.MSELoss()
        print("Pre-training stage")


    def save_when_sigint(signum, frame):
        state_dict = get_state_dict(model, optimizer, epoch)
        torch.save(state_dict, train_cfg.ckpt_dir / f"{epoch:04d}.pt")
        log_file.flush()
        log_file.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, save_when_sigint)


    if not train_cfg.ckpt_dir.exists():
        train_cfg.ckpt_dir.mkdir(parents=True)
    if not train_cfg.log_path.parent.exists():
        train_cfg.log_path.parent.mkdir(parents=True)
    log_file = open(train_cfg.log_path, "a")

    print(f"Start training from epoch {epoch}")
    for epoch in range(epoch, train_cfg.num_epochs):

        model.train()
        pbar = tqdm(train_loader)
        correct = {i: 0 for i in range(train_cfg.model.num_classes)}
        total = {i: 0 for i in range(train_cfg.model.num_classes)}
        acc_list = []
        for x, y in pbar:
            if not train_cfg.model.classifier:
                y = x.clone()

            x, y = x.to(train_cfg.device), y.to(train_cfg.device)
            y_hat = model(x)
            loss = ce_loss_fn(y_hat, y)
            pbar.set_description(f"Epoch {epoch:04d}")
            pbar.set_postfix({"loss": loss.item()})
            
            if train_cfg.model.classifier:
                for i in range(train_cfg.model.num_classes):
                    correct[i] += (y_hat.argmax(dim=1)[y == i] == i).sum().item()
                    total[i] += (y == i).sum().item()
                acc = sum(correct.values()) / sum(total.values())
                acc_list.append(acc)
                pbar.set_postfix({"acc": sum(acc_list) / len(acc_list)})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if train_cfg.model.classifier:
            for i in range(train_cfg.model.num_classes):
                correct[i] /= total[i]
            log_str = f"Epoch {epoch:04d}, Train cls acc: {cls_correct_repr(correct)}, tot acc: {sum(acc_list) / len(acc_list):.5f}"
            print(f"\033[32m{log_str}\033[0m")
            log_file.write(log_str + "\n")
        pbar.close()

        model.eval()
        with torch.no_grad():
            correct = {i: 0 for i in range(train_cfg.model.num_classes)}
            total = {i: 0 for i in range(train_cfg.model.num_classes)}
            loss = 0
            loss_cnt = 0
            for x, y in val_loader:
                if not train_cfg.model.classifier:
                    y = x.clone()
                x, y = x.to(train_cfg.device), y.to(train_cfg.device)

                if train_cfg.model.classifier:
                    y_hat = model(x)
                    y_pred = y_hat.argmax(dim=1)
                    for i in range(train_cfg.model.num_classes):
                        correct[i] += (y_pred[y == i] == i).sum().item()
                        total[i] += (y == i).sum().item()
                else:
                    y_hat = model(x)
                    loss += ce_loss_fn(y_hat, y).item()
                    loss_cnt += 1
            
            if train_cfg.model.classifier:
                acc = sum(correct.values()) / sum(total.values())
                for i in range(train_cfg.model.num_classes):
                    correct[i] /= total[i]
                log_str = f"Epoch {epoch:04d}, Valid cls acc: {cls_correct_repr(correct)}, tot acc: {acc:.5f}"
                print(f"\033[36m{log_str}\033[0m")
                log_file.write(log_str + "\n")
            else:
                log_str = f"Epoch {epoch:04d}, Valid loss: {loss / loss_cnt:.5f}"
                print(f"\033[36m{log_str}\033[0m")
                log_file.write(log_str + "\n")

    log_file.flush()
    log_file.close()

    state_dict = get_state_dict(model, optimizer, epoch)
    torch.save(state_dict, train_cfg.ckpt_dir / f"last.pt")


if __name__ == "__main__":
    main()