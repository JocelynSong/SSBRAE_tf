import configparser
import tensorflow as tf

from src.activation import Activation

__author__ = "Jocelyn"


class BRAEConfig(object):
    def __init__(self, filename):
        self._cf_parser = configparser.ConfigParser()
        self._cf_parser.read(filename)
        (self.activation_name, self.dim, self.normalize, self.weight_rec, self.weight_sem,
         self.weight_embedding, self.alpha, self.max_src_len, self.max_tar_len, self.n_epoch, self.batch_size,
         self.dropout, self.random_seed, self.min_count) = self.parse()
        self.activation = Activation(self.activation_name)
        self.optimizer_config = OptimizerConfig(filename)

    def parse(self):
        activation = self._cf_parser.get("functions", "activation")

        dim = self._cf_parser.getint("architectures", "dim")
        normalize = self._cf_parser.getboolean("architectures", "normalize")
        weight_rec = self._cf_parser.getfloat("architectures", "weight_rec")
        weight_sem = self._cf_parser.getfloat("architectures", "weight_sem")
        weight_embedding = self._cf_parser.getfloat("architectures", "weight_embedding")
        alpha = self._cf_parser.getfloat("architectures", "alpha")
        max_src_len = self._cf_parser.getint("architectures", "max_src_len")
        max_tar_len = self._cf_parser.getint("architectures", "max_tar_len")

        n_epoch = self._cf_parser.getint("parameters", "n_epoch")
        batch_size = self._cf_parser.getint("parameters", "batch_size")
        dropout = self._cf_parser.getfloat("parameters", "dropout")
        random_seed = self._cf_parser.getint("parameters", "random_seed")
        min_count = self._cf_parser.getint("parameters", "min_count")

        return (activation, dim, normalize, weight_rec, weight_sem, weight_embedding, alpha,
                max_src_len, max_tar_len, n_epoch, batch_size, dropout, random_seed, min_count)


class PARABRAEConfig(BRAEConfig):
    def __init__(self, filename):
        super(PARABRAEConfig, self).__init__(filename)
        self.beta = self._cf_parser.getfloat("architectures", "beta")


class TRANSBRAEConfig(BRAEConfig):
    def __init__(self, filename):
        super(TRANSBRAEConfig, self).__init__(filename)
        self.beta = self._cf_parser.getfloat("architectures", "beta")


class SSBRAEConfig(BRAEConfig):
    def __init__(self, filename):
        super(SSBRAEConfig, self).__init__(filename)
        self.beta = self._cf_parser.getfloat("architectures", "beta")
        self.gama = self._cf_parser.getfloat("architectures", "gama")
        self.para = self._cf_parser.getboolean("architectures", "para")
        self.trans = self._cf_parser.getboolean("architectures", "trans")
        self.para_num = self._cf_parser.getint("architectures", "para_num")
        self.trans_num = self._cf_parser.getint("architectures", "trans_num")


class OptimizerConfig(object):
    def __init__(self, filename):
        self._cf_parser = configparser.ConfigParser()
        self._cf_parser.read(filename)
        self.name, self.param = self.parse()
        self.optimizer = self.get_optimizer()

    def parse(self):
        name = self._cf_parser.get("optimizer", "optimizer")
        opt_param = self.get_opt_param(name)
        return name, opt_param

    def get_opt_param(self, optimizer):
        param = dict()
        if optimizer.lower() == "sgd":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
        elif optimizer.lower() == "sgdmomentum":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
            param["momentum"] = self._cf_parser.getfloat("optimizer", "momentum")
        elif optimizer.lower() == "adagrad":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
        elif optimizer.lower() == "adadelta":
            param["lr"] = self._cf_parser.getfloat("optimizer", "lr")
            param["decay_rate"] = self._cf_parser.getfloat("optimizer", "decay_rate")
        else:
            raise ValueError("No such optimizer:%s\n" % optimizer)
        return param

    def get_optimizer(self):
        if self.name.lower() == "sgd":
            return tf.train.GradientDescentOptimizer(self.param["lr"])
        elif self.name.lower() == "sgdmomentum":
            return tf.train.MomentumOptimizer(self.param["lr"], self.param["momentum"])
        elif self.name.lower() == "adagrad":
            return tf.train.AdagradDAOptimizer(self.param["lr"])
        elif self.name.lower() == "adadelta":
            return tf.train.AdadeltaOptimizer(self.param["lr"], self.param["decay_rate"])
        else:
            raise ValueError("No such optimizer: %s!" % self.name)









