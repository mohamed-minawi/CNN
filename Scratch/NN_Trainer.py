import numpy as np
import copy
import sys
import Load_Data as ld

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    
class NN_Trainer(object):
    def __init__(self, model,data , batch_size, num_epochs, verbose):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.verbose = verbose

        self.reset()

    def reset(self):
       
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_loss = []
        self.val_acc_history = []
        
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
          
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def step(self):
 
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        X_batch = ld.augment_batch(X_batch, 
                         rotation_range=5,
                         height_shift_range=0.16,
                         width_shift_range=0.16,
                         img_row_axis=1,
                         img_col_axis=2,
                         img_channel_axis=0,
                         horizontal_flip=True,
                         vertical_flip=False)
          
        loss = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        self.model.update_Layer()

    def train(self):
       
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)

        for t in range(num_iterations):
            self.step()

            epoch_end = (t + 1) % iterations_per_epoch == 0
            
            if epoch_end:
                self.epoch += 1
            
            progress(int(t-(iterations_per_epoch*self.epoch)), iterations_per_epoch, "Tr Loss: "+ str(round(self.loss_history[-1],5)))
         
            if  epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train, num_samples=1000)
                val_acc = self.check_accuracy(self.X_val, self.y_val)
                val_l = self.model.loss(self.X_val, self.y_val)
                self.val_loss.append(val_l)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print ("Epoch", self.epoch, "/", self.num_epochs)
                    print(" train acc ", round(train_acc,5)," train loss ", round(self.loss_history[-1],5)  ," val acc", round(val_acc,5), " val loss ", round(self.val_loss[-1],5))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
