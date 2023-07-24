import numpy as np
import matplotlib.pyplot as plt

class pyGAN:
    def __init__(self, n_inputs, gen_n_hidden_units_per_layer, dis_n_hidden_units_per_layer,  n_outputs):

        self.debug = False  

        self.create_generator(n_inputs, n_outputs, gen_n_hidden_units_per_layer)
        
        self.epochs = None

        # if isinstance(n_outputs_or_class_names, int):
        #     self.classifier = False
        #     self.n_outputs = n_outputs_or_class_names
        # else:
        #     self.classifier = True
        #     self.classes = np.array(n_outputs_or_class_names).reshape(-1,1)
        #     self.n_outputs = len(n_outputs_or_class_names)
        
        # self.weights = []
        # # Set weights using wrapper function unless there are no hidden layers. 
        # if self.n_hidden_layers == 0:
        #     self.weights = [self._make_W(self.n_inputs, self.n_outputs)]
        # else:
        #     self._weights_wrapper()
        
        if self.debug:
            for layeri, W in enumerate(self.weights):
                print(f'Layer {layeri + 1}: {W.shape=}')

        
        #IV is saved as a class variables so it can easily be shared by fprop and bprop. 
        # self.iv = None

        # self.Hs = []
        # self.mse_trace = []
        # self.percent_correct_trace = []

        # self.X_means = None
        # self.X_stds = None
        # self.T_means = None
        # self.T_stds = None

    def create_generator(self, n_inputs, n_outputs, hiddens):
        self.gen_hiddens = hiddens
        self.gen_n_hidden_layers = len(hiddens)

        self.gen_n_inputs = n_inputs
        self.gen_n_outputs = n_outputs

        self.gen_Ws = []
        if self.gen_n_hidden_layers == 0:
            self.gen_Ws = [self._make_W(self.gen_n_inputs, self.gen_n_outputs)]
        else:
            self.gen_Ws = self._weights_wrapper(self.gen_n_inputs, self.gen_hiddens, self.gen_n_outputs)
        
        self.gen_Hs = []
        self.gen_mse_trace = []
        
        self.gen_X_means = None
        self.gen_X_stds = None
        self.gen_T_means = None
        self.gen_T_stds = None

    def create_discriminator(self, n_inputs, n_outputs, hiddens):
        self.dis_hiddens = hiddens
        self.dis_n_hidden_layers = len(hiddens)

        self.dis_n_inputs = n_inputs
        self.dis_n_outputs = n_outputs

        self.dis_Ws = []
        if self.dis_n_hidden_layers == 0:
            self.dis_Ws = [self._make_W(self.dis_n_inputs, self.dis_n_outputs)]
        else:
            self.dis_Ws = self._weights_wrapper(self.dis_n_inputs, self.dis_hiddens, self.dis_n_outputs)

        self.dis_Hs = []
        self.dis_mse_trace = []

        self.dis_X_means = None
        self.dis_X_stds = None
        self.dis_T_means = None
        self.dis_T_stds = None

        self.dis_iv = None


    def __repr__(self):
        s = f'NeuralNetwork({self.n_inputs}, {self.hiddens}, '
        if self.classifier:
            s += f'{self.classes})'
            kind = 'classification'
        else:
            s += f'{self.n_outputs})'
            kind = 'regression'
        if self.epochs == 0:
            s += f'\n Not trained yet on a {kind} problem.'
        else:
            s += f'\n Trained on a {kind} problem for {self.epochs} epochs '
            if self.classifier:
                s += f'with a final training percent correct of {self.percent_correct_trace[-1]:.2f}.'
            else:
                s += f'with a final training MSE of {self.mse_trace[-1]:.4g}.'
        return s

    def __str__(self):
        return self.__repr__()
    
    def set_debug(self, true_false):
        self.debug = true_false
    
    def _weights_wrapper(self, n_inputs, Hs, n_outputs):
        Ws = []
        Ws.append(self._make_W(n_inputs, Hs[0]))
        for i in range(len(Hs) - 1):
            Ws.append(self._make_W(Hs[i], Hs[i+1]))
        Ws.append(self._make_W(Hs[-1],n_outputs))
        return Ws

    def _make_W(self, ni, nu):
        return np.random.uniform(-1, 1, size=(ni + 1, nu)) / np.sqrt(ni + 1)

    def train(self, X, T, n_epochs, learning_rate):
        if self.debug:
            print('----------Starting train()')
            
        learning_rate = learning_rate / X.shape[0]

        if self.debug:
            print(f'Adjusted {learning_rate=}')
            

        X = self._standardizeX(X)

        for epoch in range(n_epochs):
            #Forward Prop
            gen_Y = self._fprop(X, type="G")
            dis_Y = self._fprop(gen_Y, type="D")

            self._bprop(X, dis_Y, learning_rate, target=self._standardizeT(T))

            if self.classifier:
                self.mse_trace.append(self._E(X, self.iv))
                self.percent_correct_trace.append(self.percent_correct(T, Y_classes))
            else:
                self.mse_trace.append(self._E(X, T))
        self.epochs = n_epochs

    def use(self, X, standardized=False):
        if not standardized:
            X = self._standardizeX(X)

        if self.classifier:
            Y_classes, Y_softmax = self._fprop(X)
            return Y_classes, Y_softmax
        else:
            Y = self._fprop(X)
            return self._unstandardizeT(Y)


    def _fprop(self, X, type):

        if type == "G":
            Hs = self.gen_Hs
            Ws = self.gen_Ws            
        else:
            Hs = self.dis_Hs
            Ws = self.dis_Ws

        Hs = [X]
        Hs.append(self._f(self._add_ones(X) @ Ws[0]))

        for i in range(1,len(self.weights)-1):
            Hs.append(self._f(self._add_ones(Hs[-1]) @ Ws[i]))

        #Takes care of edge case of 0 hidden layers. 
        # if self.n_hidden_layers == 0:
        #     Y = self.Hs[-1]
        # else:
        Y = self._add_ones(Hs[-1]) @ Ws[-1]

        if type == "G":
            self.gen_Hs = Hs
        else:   
            self.dis_Hs = Hs

        if type == "D":
            Y_softmax = self._softmax(Y)
            Y_classes = self.classes[np.argmax(Y_softmax, axis=1)]
            return Y_classes, Y_softmax
        else:
            return Y

    def _bprop(self, X, Y, learning_rate, prop_gen=False, X_gen = None):
        delta = -2 * (self.iv - Y)
        deltai = -0.5 * np.power((self.iv - Y), -0.5)
        #delta = -2 * (target - Y)
            
        for i in range(len(self.dis_Ws)-1, 0, -1):
            self.dis_Ws[i] -= learning_rate * self._add_ones(self.dis_Hs[i]).T @ delta
            delta = delta @ self.dis_Ws[i][1:, :].T * self._df(self.dis_Hs[i])
            
            if prop_gen:
                deltai = deltai @ self.dis_Ws[i][1:, :].T * self._df(self.dis_Hs[i])
        self.dis_Ws[0] -= learning_rate * self._add_ones(X).T @ delta
        
        if prop_gen:
            self._bprop_gen(X_gen, Y, deltai, learning_rate)


    def _bprop_gen(self, X, Y, deltai, learning_rate):
        
        for i in range(len(self.gen_Ws)-1, 0, -1):
            self.gen_Ws[i] -= learning_rate * self._add_ones(self.gen_Hs[i]).T @ deltai
            deltai = deltai @ self.gen_Ws[i][1:, :].T * self._df(self.gen_Hs[i])

        self.gen_Ws[0] -= learning_rate * self._add_ones(X).T @ deltai


    def _standardizeX(self, X):        
        if self.X_means is None:
            self.X_means = np.mean(X, axis=0)
            self.X_stds = np.std(X, axis=0)
            self.X_stds[self.X_stds == 0] = 1
        return (X - self.X_means) / self.X_stds
        
    def _standardizeT(self, T):
        # return T
        if self.T_means is None:
            self.T_means = np.mean(T, axis=0)
            self.T_stds = np.std(T, axis=0)
            self.T_stds[self.T_stds == 0] = 1
        return (T - self.T_means) / self.T_stds

    def _unstandardizeT(self, T):
        # return T
        
        if self.T_means is None:
            raise Exception('T not standardized yet')

        return (T * self.T_stds) + self.T_means
    
    def _E(self, X, T_iv_or_T):
        if self.classifier:
            Y_class_names, Y_softmax = self.use(X, standardized=True)
            sq_diffs = (T_iv_or_T - Y_softmax) ** 2
        else:
            Y = self.use(X, standardized=True)
            sq_diffs = (T_iv_or_T - Y) ** 2
        return np.mean(sq_diffs)
    
    def _add_ones(self, M):
        return np.insert(M, 0, 1, 1)

    def _make_indicator_vars(self, T):
        return (T == np.unique(T)).astype(int)

    def _softmax(self, Y):
        fs = np.exp(Y)  # N x K
        denom = np.sum(fs, axis=1).reshape((-1, 1))
        return fs / denom

    def _f(self, S):
        return np.tanh(S)

    def _df(self, fS):
        return (1 - fS ** 2)

    def percent_correct(self, T, Y_classes):
        return 100 * np.mean(T == Y_classes)

    def plot_mse_trace(self):
        if len(self.mse_trace) == 0:
            print("Train Model Before Attempting to Plot!")
            return None

        plt.plot(self.mse_trace)
        plt.title("MSE Trace")
        plt.xlabel("Epoch #")
        plt.ylabel("MSE")

    def plot_percent_correct_trace(self):
        if len(self.percent_correct_trace) == 0:
            print("Train Model Before Attempting to Plot!")
            return None

        plt.plot(self.percent_correct_trace)
        plt.title("% Correct Trace")
        plt.xlabel("Epoch #")
        plt.ylabel("% Correct")