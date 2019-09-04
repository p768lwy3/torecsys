from trainer import Trainer


class RankingTrainer(Trainer):
    def __init__(self,
                 inputs_wrapper : _Inputs,
                 model          : _Model,
                 neg_sampler    : callable,
                 regularizer    : Regularizer = Regularizer(0.1, 2),
                 loss           : type = nn.MSELoss,
                 optimizer      : type = optim.AdamW,
                 epochs         : int = 10,
                 verboses       : int = 2,
                 log_step       : int = 500,
                 log_dir        : str = "logdir", 
                 use_jit        : bool = False,
                 **kwargs):
        # refer to parent class
        super(RankingTrainer, self).__init__(
            inputs_wrapper = inputs_wrapper,
            model          = model,
            regularizer    = regularizer,
            labels_name    = None,
            loss           = loss,
            optimizer      = optimizer,
            epochs         = epochs,
            verboses       = verboses,
            log_step       = log_step,
            log_dir        = log_dir,
            use_jit        = use_jit,
            **kwargs
        )

        # bind neg_sampler to neg_sampler
        self.neg_sampler = neg_sampler

    def _iterate(self, 
                 pos_inputs: Dict[str, torch.Tensor], 
                 neg_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # set batch size and number of negative samples for reshaping
        batch_size = pos_inputs.size(0)
        num_samples = neg_inputs.size(0)

        # cat pos_inputs and neg_inputs into a dictionary of tensors - batch_inputs
        batch_inputs = dict()
        for inp_name in pos_inputs:
            num_fields = neg_inputs[inp_name].size(1)
            embed_size = neg_inputs[inp_name].size(2)
            pos_tensors = pos_inputs[inp_name].squeeze(1)
            neg_tensors = neg_inputs[inp_name].view(
                batch_size, num_samples, num_fields, embed_size)
            batch_inputs[inp_name] = torch.cat((pos_tensors, neg_tensors), dim=1)
        
        # calculate forward prediction, where outputs' shape = (B * (1 + Nneg), 1)
        outputs = self.sequential(batch_inputs)

        # split outputs into pos_outputs and neg_outputs
        pos_outputs, neg_outputs = outputs.view(batch_size, -1, 1).split(
            (1, num_samples), dim=1)
        
        # calculate loss and regularized loss
        loss = self.loss(pos_outputs, neg_outputs)
        if self.regularizer is not None:
            named_params = list(self.sequential.named_parameters())
            reg_loss = self.regularizer(named_params)
            loss += reg_loss

        # calculate backward and optimize 
        loss.backward()
        self.optimizer.step()

        # return loss to log stream and tensorboard
        return loss
    
    def fit(self, dataloader: torch.utils.data.DataLoader):
        # initialize global_step = 0 for logging
        global_step = 0

        # number of batches
        num_batch = len(dataloader)

        # loop through n epochs
        for epoch in range(self.epochs):
            # initialize loss variables to store aggregated loss
            steps_loss = 0.0
            epoch_loss = 0.0

            # logging of the epoch
            if self.verboses >= 1:
                self.logger.info("Epoch %s / %s:" % (epoch + 1, self.epochs))
            
            # initialize progress bar of dataloader of this epoch
            pbar = tqdm(dataloader, desc="step loss : ??.????")

            for i, pos_inputs in enumerate(pbar):
                # generate negative samples
                neg_inputs = dict()
                """
                <!--- code --->
                """
                
                # iteration of the batch
                loss = self._iterate(pos_inputs, neg_inputs)
                
                # add step loss to steps_loss and epoch_loss
                loss_val = loss.item()
                steps_loss += loss_val
                epoch_loss += loss_val

                # set loss to the description of pbar
                pbar.set_description("step loss : %.4f" % (loss_val))

                # log for each y steps
                if global_step % self.log_step == 0:
                    if self.verboses >= 1:
                        self.logger.debug("step avg loss at step %d of epoch %d : %.4f" % (i, epoch, steps_loss / self.log_step))
                    if self.verboses >= 2:
                        self.writer.add_scalar("training/steps_avg_loss", steps_loss / self.log_step, global_step=global_step)    
                    steps_loss = 0.0

                global_step += 1

            # log for each epoch
            if self.verboses >= 1:
                self.logger.info("epoch avg loss : %.4f" % (epoch_loss / num_batch))
            
            if self.verboses >= 2:
                self.writer.add_scalar("training/epoch_avg_loss", epoch_loss / num_batch, global_step=epoch)

    def evalaute(self):
        return
        