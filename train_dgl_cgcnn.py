import dgl.function as fn
import torch 
import dgl 
import torch.nn.functional as F
import pickle 
import numpy as np
import time
import os
from collections import defaultdict
from cgcnn.dgl_data import CrystalLoader, AverageMeter, collate_crystal_graphs_for_regression, save_best, save_checkpoint
from cgcnn.dgl_model import CGConvNet


def main():
    data_train = CrystalLoader('/scratch/westgroup/mpnn/gasdb_dgl_graphs/gaspy_splits/gaspy_train.pkl')
    data_valid = CrystalLoader('/scratch/westgroup/mpnn/gasdb_dgl_graphs/gaspy_splits/gaspy_valid.pkl')
    data_test = CrystalLoader('/scratch/westgroup/mpnn/gasdb_dgl_graphs/gaspy_splits/gaspy_test.pkl')

    print(f"Sizes of train, valid and test sets: {len(data_train)}, {len(data_valid)}, {len(data_test)}")

    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=20,
        collate_fn=collate_crystal_graphs_for_regression,
        num_workers=8)

    valid_loader = torch.utils.data.DataLoader(
        data_valid,
        batch_size=20,
        collate_fn=collate_crystal_graphs_for_regression,
        num_workers=8)

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=20,
        collate_fn=collate_crystal_graphs_for_regression,
        num_workers=8)

    print("\t Creating Model and params")
    model = CGConvNet()
    criterion = torch.nn.MSELoss()
    evaluation = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if torch.cuda.is_available():
        model =model.to('cuda')
        criterion = criterion.to('cuda')
        evaluation = evaluation.to('cuda')
    
    
    checkpoint_dir = '/scratch/westgroup/mpnn/gasdb_dgl_graphs/gaspy_splits/run_001'
    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
    start_epoch = 0
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.isfile(best_model_file):
        print("=> loading best model '{}'".format(
            best_model_file)) 
        checkpoint = torch.load(best_model_file)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_er1']
        print("\t best MAE is {}".format(best_acc1))
        model.load_state_dict(checkpoint['state_dict'])
        print(
            "=> loaded best model '{}' (epoch {})".format(
                best_model_file,
                checkpoint['epoch'])) 
    else:
        print("=> no best model found at '{}'".format(
            best_model_file)) 
    loss_dict = defaultdict(list)
    best_er1 = 0
    for epoch in range(start_epoch, 540):
        train_loss, train_error = train(
            train_loader, model, criterion, optimizer, epoch, evaluation)
        val_loss, er1 = eval(
            valid_loader, model, epoch,criterion, evaluation,mode='Valid')
        
        test_loss, test_error = eval(
            test_loader, model, epoch,criterion,evaluation,mode='Test')
        loss_dict['train_loss'].append(train_loss)
        loss_dict['val_loss'].append(val_loss)
        loss_dict['test_loss'].append(test_loss)
        loss_dict['train_error'].append(train_error)
        loss_dict['val_error'].append(er1)
        loss_dict['test_error'].append(test_error)
        checkpoint_dir = '/scratch/westgroup/mpnn/gasdb_dgl_graphs/gaspy_splits/run_001'
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, 'metrics_dict.pkl'), 'wb') as outfile:
            pickle.dump(loss_dict, outfile)
        if not best_er1 or er1 < best_er1:
            is_best = True
            best_er1 = er1
            save_best({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_er1': best_er1,
                             'er1': er1,
                             'optimizer': optimizer.state_dict()},
                            directory=checkpoint_dir,
                            )
        save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'best_er1': best_er1,
                               'er1': er1,
                               'optimizer': optimizer.state_dict(),
                               },
                              directory=checkpoint_dir,
                              )
def train(train_loader, model, criterion, optimizer, epoch, evaluation):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()
    total_loss = 0
    total_error = 0

    # switch to train mode
    model.train()

    end = time.time()
    for batch_id, batch_data in enumerate(train_loader):
        # Prepare input data
        if torch.cuda.is_available():
            bg, labels = batch_data
            bg.to(torch.device('cuda'))
            labels = labels.to('cuda')
       
        # Measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        # Compute output
        output = model(bg)
        train_loss = criterion(output, labels)
        train_eval = evaluation(output, labels)
        # Logs
        losses.update(criterion(output, labels).item(), bg.batch_size)
        error_ratio.update((evaluation(output, labels)).item(), bg.batch_size)
        total_loss += train_loss*bg.batch_size
        total_error += train_eval*bg.batch_size
        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % 20 == 0 and batch_id > 0:

            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Error Ratio {err.val:.4f} ({err.avg:.4f}) \t' .format(
                    epoch,
                    batch_id,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    err=error_ratio))
    total_loss = total_loss /len(train_loader.dataset)
    total_error = total_error / len(train_loader.dataset)
    print('epoch {:d}/{:d}, training loss {:.4f}, training score {:.4f}'.format(
        epoch + 1, 100, total_loss, total_error))
    print(os.system("nvidia-smi"))

    print('Epoch: [{0}] Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time))
    return total_loss.detach().cpu().numpy(), total_error.detach().cpu().numpy()


def eval(eval_loader, model, epoch, criterion,evaluation,mode='Valid'):

    batch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()
    total_loss,total_error = 0, 0
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_id, batch_data in enumerate(eval_loader):
        # Prepare input data
        if torch.cuda.is_available():
            bg, labels = batch_data
            bg.to(torch.device('cuda'))
            labels = labels.to('cuda')
       
        # Compute output
        with torch.no_grad():
            output = model(bg)
        eval_loss = criterion(output,labels)
        eval_error = evaluation(output,labels)
        # Logs
        losses.update(eval_loss.item(), bg.batch_size)
        error_ratio.update(eval_error.item(), bg.batch_size)
        total_loss += eval_loss*bg.batch_size
        total_error += eval_error*bg.batch_size
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_id % 20 == 0 and batch_id > 0:
            print(
                '{}: [{}/{}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Erro Ratio {err.val:.4f} ({err.avg:.4f}) \t' .format(mode,
                    batch_id,
                    len(eval_loader),
                    batch_time=batch_time,
                    loss=losses,
                    err=error_ratio)) 
    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses)) 
    total_loss = total_loss / len(eval_loader.dataset)
    total_error = total_error/len(eval_loader.dataset)
    print('epoch {:d}/{:d}, {} loss {:.4f}, {} score {:.4f}'.format(
        epoch + 1, 540, mode,total_loss, mode, total_error))
    return total_loss.detach().cpu().numpy(), total_error.detach().cpu().numpy()





if __name__ == '__main__':
    main()