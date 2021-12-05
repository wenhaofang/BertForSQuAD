import os
import tqdm
import numpy as np
import torch

def save_checkpoint(save_path, model, optim, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']

def qa_train(dataloader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    for mini_batch in tqdm.tqdm(dataloader):
        mini_batch = [batch_data_item.to(device) for batch_data_item in mini_batch]
        input_ids, input_mask, segment_ids, start_pos, end_pos, _ = mini_batch
        output = model(input_ids, input_mask, segment_ids)
        start_logits, end_logits = output.start_logits, output.end_logits
        total_loss = (
            criterion(start_logits, start_pos) +
            criterion(end_logits, end_pos)
        ) / 2
        train_loss += total_loss.item()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return {
        'loss': train_loss / len(dataloader)
    }

def qa_valid(dataloader, model, criterion, optimizer, device):
    model.eval()
    valid_loss = 0
    start_true_fold = []
    start_prob_fold = []
    end_true_fold = []
    end_prob_fold = []
    with torch.no_grad():
        for mini_batch in tqdm.tqdm(dataloader):
            mini_batch = [batch_data_item.to(device) for batch_data_item in mini_batch]
            input_ids, input_mask, segment_ids, start_pos, end_pos, _ = mini_batch
            output = model(input_ids, input_mask, segment_ids)
            start_logits, end_logits = output.start_logits, output.end_logits
            total_loss = (
                criterion(start_logits, start_pos) +
                criterion(end_logits, end_pos)
            ) / 2
            valid_loss += total_loss.item()
            start_true_fold.append(start_pos)
            start_prob_fold.append(start_logits)
            end_true_fold.append(end_pos)
            end_prob_fold.append(end_logits)
    start_true_fold = torch.cat(start_true_fold)
    start_prob_fold = torch.cat(start_prob_fold)
    start_pred_fold = start_prob_fold.argmax(dim = -1)
    end_true_fold = torch.cat(end_true_fold)
    end_prob_fold = torch.cat(end_prob_fold)
    end_pred_fold = end_prob_fold.argmax(dim = -1)
    return {
        'loss': valid_loss / len(dataloader),
        'start_true_fold': start_true_fold,
        'start_prob_fold': start_prob_fold,
        'start_pred_fold': start_pred_fold,
        'end_true_fold': end_true_fold,
        'end_prob_fold': end_prob_fold,
        'end_pred_fold': end_pred_fold
    }

def qa_save_sample(
    save_folder,
    start_true_fold,
    start_prob_fold,
    start_pred_fold,
    end_true_fold,
    end_prob_fold,
    end_pred_fold
):
    start_true_path = os.path.join(save_folder, 'start_true_fold.txt')
    start_prob_path = os.path.join(save_folder, 'start_prob_fold.txt')
    start_pred_path = os.path.join(save_folder, 'start_pred_fold.txt')
    end_true_path = os.path.join(save_folder, 'end_true_fold.txt')
    end_prob_path = os.path.join(save_folder, 'end_prob_fold.txt')
    end_pred_path = os.path.join(save_folder, 'end_pred_fold.txt')
    np.savetxt(start_true_path, start_true_fold)
    np.savetxt(start_prob_path, start_prob_fold)
    np.savetxt(start_pred_path, start_pred_fold)
    np.savetxt(end_true_path, end_true_fold)
    np.savetxt(end_prob_path, end_prob_fold)
    np.savetxt(end_pred_path, end_pred_fold)
