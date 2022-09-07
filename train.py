"""General training pipeline from YAML config"""

import os
from tqdm import tqdm
import torch
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)

def train(config,
          config_id,
          model,
          device,
          optimizer,
          scheduler,
          loss_function,
          val_metric,
          train_loader,
          val_loader,
          epoch_range,
          logger,
          writer,
          checkpoint_dir,
          model_dir):
    
    model.train()
    # Training protocol
    best_metric = -1
    best_metric_epoch = -1
    start_epoch, epochs = epoch_range
    global_step = 0
    for epoch in range(start_epoch, epochs):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{epochs}")
        
        epoch_loss = 0
        step = 0
        for batch in tqdm(train_loader):
            step += 1
            optimizer.zero_grad()
            inputs, labels = (
                batch["image"].to(device),
                batch["label"].to(device),
            )

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            scheduler.step()
            # print(optimizer.param_groups[0]['lr'])

            # logging
            writer.add_scalar('Loss/train', loss.data, global_step)

            # validate
            if (global_step) % config["val_interval"] == 0:
                val_dice = validate(config, model, device, loss_function,
                                    val_metric, val_loader, writer, global_step)
                # track best model
                if val_dice > best_metric:
                    best_metric = val_dice
                    best_metric_epoch = epoch + 1

                    torch.save(model.state_dict(), os.path.join(model_dir, f"{config_id}_best_model.pth"))
                    print("Saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {val_dice:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
            global_step += 1

        # logging
        # logger.log("loss", (epoch + 1, epoch_loss))
        # logger.log("dice", (epoch + 1, epoch_dice))

        # writer.add_scalar('Dice/train', epoch_dice, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if config["checkpoint_interval"]:
            if (epoch + 1) % config["checkpoint_interval"] == 0:
                # save model at every checkpoint interval

                checkpoint_path = os.path.join(checkpoint_dir, f"epoch{epoch + 1}.tar")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                }, checkpoint_path)

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

def validate(config, model, device, loss_function,
             val_metric, val_loader, writer, global_step):
    model.eval()
    val_loss = 0
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=6)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=6)])
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = config["crop_shape"]
            sw_batch_size = 4
            # Run over the input image with a sliding window, run inference on each fragment and then aggregate to get the overall result.
            val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
            loss = loss_function(val_outputs, val_labels)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]

            # compute loss and dice for current iter
            val_loss += loss.item()
            val_metric(y_pred=val_outputs, y=val_labels)

    # total loss and dice over validation set
    val_loss /= len(val_loader.dataset)
    val_dice = val_metric.aggregate().item()
    val_metric.reset()
    writer.add_scalar('Loss/val', val_loss, global_step)
    writer.add_scalar('Dice/val', val_dice, global_step)

    return val_dice