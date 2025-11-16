class Core:
    def wrapped_model(self, model):
        '''
        Pack model for inference
        '''
        raise NotImplementedError
    
    def loss_func(self, model, data):
        '''
        Loss function for training
        '''
        raise NotImplementedError

    def on_validation_step(self, model, data):
        pass