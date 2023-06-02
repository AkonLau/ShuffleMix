import torch
import torch.nn as nn
import numpy as np

class ShuffleMix(nn.Module):
    def __init__(self, model, conf):
        super().__init__()
        self.model = model
        self.alpha = conf.alpha
        self.ratio = conf.ratio
        self.mix_type = conf.mix_type
        self.index_type = conf.index_type
        self.k_layer1 = conf.k_layer1
        self.k_layer2 = conf.k_layer2
        self.ada_dropout = conf.ada_dropout

        self.add_noise_level = conf.add_noise_level
        self.mult_noise_level = conf.mult_noise_level

        self.lam = None
        self.module_list = []
        for n, m in self.model.named_modules():
            if n[:-1]=='layer':
                self.module_list.append(m)
        print('| Length(self.module_list)', len(self.module_list))
        assert self.k_layer1 < self.k_layer2 and self.k_layer2 <= len(self.module_list)

    def forward(self, x, target=None):
        if target==None:
            out = self.model(x)
            return out
        else:
            if self.mix_type == 'soft':
                self.lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0.0 else 1.0
            elif self.mix_type == 'hard':
                self.lam = 0 # hard mix means replacing the processing part completely
            else:
                raise ValueError("mix_type must be \'hard\' or \'soft\'.")

            k = np.random.randint(self.k_layer1, self.k_layer2)
            self.indices = torch.randperm(target.size(0)).cuda()

            if k == -1: # input mix
                cx = int(x.size(1) * self.ratio)
                cx = cx if cx > 0 else 1

                if self.index_type == 'random':
                    cx_index = np.random.choice(x.size(1), cx, replace=False)
                    if self.ada_dropout:
                        x[:, cx_index] = x[:, cx_index] * 0; self.lam = 1
                    else:
                        x[:, cx_index] = x[:, cx_index] * self.lam + x[self.indices][:, cx_index] * (1-self.lam)
                else:
                    x[:, :cx] = x[:, :cx] * self.lam + x[self.indices][:, :cx] * (1-self.lam)
                x = self._noise(x, self.add_noise_level, self.mult_noise_level)
                out = self.model(x)
                self.lam = 1 - cx / x.size(1) * (1 - self.lam)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()

            return out, target, target[self.indices], self.lam

    def hook_modify(self, module, input, output):
        cx = int(output.size(1) * self.ratio)
        if self.index_type == 'random':
            cx_index = np.random.choice(output.size(1), cx, replace=False)
            if self.ada_dropout:
                output[:, cx_index] = output[:, cx_index] * 0; self.lam = 1
            else:
                output[:, cx_index] = output[:, cx_index] * self.lam + output[self.indices][:,cx_index] * (1-self.lam)
        else:
            output[:, :cx] = output[:, :cx] * self.lam + output[self.indices][:, :cx] * (1-self.lam)
        self.lam = 1 - cx / output.size(1) * (1 - self.lam)
        output = self._noise(output, self.add_noise_level, self.mult_noise_level)
        return output

    def _noise(self, x, add_noise_level=0.0, mult_noise_level=0.0):
        add_noise = 0.0
        mult_noise = 1.0
        with torch.cuda.device(0):
            if add_noise_level > 0.0:
                add_noise = add_noise_level * np.random.beta(2, 5) * torch.cuda.FloatTensor(x.shape).normal_()
            if mult_noise_level > 0.0:
                mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.cuda.FloatTensor(x.shape).uniform_()-1) + 1
        return mult_noise * x + add_noise

class ManifoldMixup(nn.Module):
    def __init__(self, model, conf):
        super().__init__()
        self.model = model
        self.alpha = conf.alpha
        self.add_noise_level = conf.add_noise_level
        self.mult_noise_level = conf.mult_noise_level

        self.lam = None
        self.module_list = []
        for n, m in self.model.named_modules():
            if n[:-1]=='layer':
                self.module_list.append(m)

    def forward(self, x, target=None):
        if target==None:
            out = self.model(x)
            return out
        else:
            self.lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0.0 else 1.0
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            if k == -1: # vanilla mixup
                x = x * self.lam + x[self.indices] * (1 - self.lam)
                x = self._noise(x, self.add_noise_level, self.mult_noise_level)
                out = self.model(x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
            return out, target, target[self.indices], self.lam

    def hook_modify(self, module, input, output):
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        output = self._noise(output, self.add_noise_level, self.mult_noise_level)
        return output

    def _noise(self, x, add_noise_level=0.0, mult_noise_level=0.0):
        add_noise = 0.0
        mult_noise = 1.0
        with torch.cuda.device(0):
            if add_noise_level > 0.0:
                add_noise = add_noise_level * np.random.beta(2, 5) * torch.cuda.FloatTensor(x.shape).normal_()
            if mult_noise_level > 0.0:
                mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.cuda.FloatTensor(x.shape).uniform_()-1) + 1
        return mult_noise * x + add_noise