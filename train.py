# -- coding: utf-8 --
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
# from util.visualizer import Visualizer
from util import util
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt) 
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    # visualizer = Visualizer(opt)
    total_steps = 0
    writer = SummaryWriter('GFRNet_FlowNet')
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize #

            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visuals = model.get_current_visuals()
                losses = model.get_current_losses()
                util.display_current_results(writer,visuals,losses,total_steps,save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                util.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')
                #exit('sss')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        print('#####################################')
        model.update_learning_rate()
    writer.close()