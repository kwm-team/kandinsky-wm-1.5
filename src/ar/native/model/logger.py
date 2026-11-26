import functools
import psutil
import torch
from torch import nn


def get_full_tensor(tensor, emergency=False):
    if hasattr(tensor, "full_tensor") and not emergency:
        return tensor.full_tensor()
    return tensor


class ModelLogger:
    USE_EXTENDED_LOGS = False
    _psutil_warning_emitted = False

    @classmethod
    def set_extended_logs(cls, value=True):
        cls.USE_EXTENDED_LOGS = value

    @staticmethod
    def log_transformer(cls):
        def log_for_init(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                self.log_dict = {}
                return result
            return wrapper

        def collate_log(self, log_dict):
            for module_name, module in self.__dict__['_modules'].items():
                if isinstance(module, nn.ModuleList):
                    log_dict.update(collate_log(module, log_dict))
                elif getattr(module, "get_logs", None) is not None:
                    for log_name, log_value in module.get_logs().items():
                        log_dict.setdefault(log_name, []).append(log_value)
            return log_dict

        def log_for_forward(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                self.log_dict = collate_log(self, {})
                return result
            return wrapper

        def wrap(cls):
            def get_logs(self):
                log_dict = {f'model_status/{key}': value for key, value in self.log_dict.items()}
                keys = list(log_dict.keys())
                for key in keys:
                    values = torch.stack(log_dict[key])
                    if key.endswith('max'):
                        log_dict[key] = values.max().cpu().item()
                    elif key.endswith(('mean', 'std')):
                        log_dict[key] = values.mean().cpu().item()
                    elif key.endswith('min'):
                        log_dict[key] = values.min().cpu().item()
                    elif key.endswith('#'):
                        for i, v in enumerate(values):
                            log_dict[key+f'{i}'] = v.cpu().item()
                        if key in log_dict:
                            del log_dict[key]
                    else:
                        log_dict[key] = values
                return log_dict
            cls.__init__ = log_for_init(cls.__init__)
            cls.forward = log_for_forward(cls.forward)
            cls.get_logs = get_logs
            return cls

        return wrap(cls)

    @staticmethod
    def log_transformer_block(cls):
        def log_for_init(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                self.log_dict = {}
                return result
            return wrapper

        def log_for_forward(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                visual_result = func(self, *args, **kwargs)
                for module_name, module in self.__dict__['_modules'].items():
                    if getattr(module, "get_logs", None) is not None:
                        for log_name, log_value in module.get_logs().items():
                            self.log_dict[f'{module_name}_{log_name}'] = log_value
                visual_norm = torch.norm(visual_result.detach(), dim=-1)
                self.log_dict[f'visual_norm_mean'] = visual_norm.mean()
                self.log_dict[f'visual_norm_max'] = visual_norm.mean()

                # log hidden states blockwise
                if ModelLogger.USE_EXTENDED_LOGS:
                    visual_abs = visual_result.detach().abs()
                    # end name with '#' for layerwise logging
                    self.log_dict[f'visual_abs_mean#'] = visual_abs.mean(dim=-1).mean()
                    self.log_dict[f'visual_norm_mean#'] = visual_norm.mean()
                return visual_result
            return wrapper

        def wrap(cls):
            def get_logs(self):
                return self.log_dict.copy()
            cls.__init__ = log_for_init(cls.__init__)
            cls.forward = log_for_forward(cls.forward)
            cls.get_logs = get_logs
            return cls
        return wrap(cls)

    @staticmethod
    def log_attention(cls):
        def log_for_init(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result = func(self, *args, **kwargs)
                self.log_dict = {}
                return result
            return wrapper

        def log_qkv(func, layerwise=False):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                q, k, v = func(self, *args, **kwargs)
                if ModelLogger.USE_EXTENDED_LOGS:
                    q_norm, k_norm, v_norm = (
                        torch.norm(q.detach(), dim=-1),
                        torch.norm(k.detach(), dim=-1),
                        torch.norm(v.detach(), dim=-1)
                    )
                    postfix = "#" if layerwise else ""
                    self.log_dict['q_norm_mean'+postfix] = q_norm.mean()
                    self.log_dict['k_norm_mean'+postfix] = k_norm.mean()
                    self.log_dict['v_norm_mean'+postfix] = v_norm.mean()
                    # self.log_dict['q_norm_std'+postfix] = q_norm.std()
                    # self.log_dict['k_norm_std'+postfix] = k_norm.std()
                    # self.log_dict['v_norm_std'+postfix] = v_norm.std()
                return q, k, v
            return wrapper

        def log_before_norm_qk(func, layerwise=False):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                q, k = func(self, *args, **kwargs)
                if ModelLogger.USE_EXTENDED_LOGS:
                    q_before, k_before = args[:2]
                    q_norm, k_norm = (
                        torch.norm(q_before.detach(), dim=-1),
                        torch.norm(k_before.detach(), dim=-1),
                    )
                    postfix = "#" if layerwise else ""
                    self.log_dict['q_before_rms_norm_mean'+postfix] = q_norm.mean()
                    self.log_dict['k_before_rms_norm_mean'+postfix] = k_norm.mean()
                return q, k
            return wrapper

        def log_sdpa(func, layerwise=False):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result, softmax_lse, _ = func(self, *args, return_attn_probs=True, **kwargs)
                softmax_lse = softmax_lse.detach()
                postfix = "#" if layerwise else ""
                self.log_dict['softmax_norm_max'+postfix] = softmax_lse.max()
                self.log_dict['softmax_norm_mean'+postfix] = softmax_lse.mean()
                self.log_dict['softmax_norm_min'+postfix] = softmax_lse.min()
                return result
            return wrapper

        def log_for_sparsity(func, layerwise=False):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                result, sparsity = func(self, *args, return_sparsity=True, **kwargs)
                postfix = "#" if layerwise else ""
                self.log_dict['block_mask_sparsity_mean'+postfix] = torch.tensor(sparsity, dtype=torch.float32)
                return result
            return wrapper

        def wrap(cls):
            def get_logs(self):
                return self.log_dict.copy()
            cls.__init__ = log_for_init(cls.__init__)

            if hasattr(cls, "scaled_dot_product_attention"):
                cls.scaled_dot_product_attention = log_sdpa(cls.scaled_dot_product_attention, layerwise=False)

            if hasattr(cls, "attention_flex"):
                cls.attention_flex = log_for_sparsity(cls.attention_flex, layerwise=False)

            if hasattr(cls, 'get_qkv'):
                cls.get_qkv = log_qkv(cls.get_qkv, layerwise=True)

            if hasattr(cls, 'norm_qk'):
                cls.norm_qk = log_before_norm_qk(cls.norm_qk, layerwise=True)

            cls.get_logs = get_logs
            return cls
        return wrap(cls)

    @staticmethod
    def log_model_status(model, emergency=False):
        '''
        Args:
            emergency - disable distributed ops if True
        '''
        params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        grads = [p.grad for p in model.parameters() if p.grad is not None and p.requires_grad]
        # NOTE: in case of exception triggered on NaN in backward, these objects
        # can become usual Tensor instead of DTensor
        weights_norm = get_full_tensor(nn.utils.get_total_norm(params), emergency).cpu().item()
        grad_norm = get_full_tensor(nn.utils.get_total_norm(grads), emergency).cpu().item()
        log_dict = model.get_logs()
        log_dict.update({
            "model_status/weights_norm": weights_norm,
            "model_status/grad_norm": grad_norm,
        })
        if ModelLogger.USE_EXTENDED_LOGS:
            log_dict.update({
                f"weights_norm/{n}": get_full_tensor(nn.utils.get_total_norm(p), emergency).cpu().item()
                for n, p in model.named_parameters() if p.requires_grad
            })
        return log_dict

    @classmethod
    def _warn_system_metrics_unavailable(cls, exc):
        if cls._psutil_warning_emitted:
            return
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        ):
            return
        print(f"[logger] system metrics unavailable: {exc}", flush=True)
        cls._psutil_warning_emitted = True

    @classmethod
    def log_system_status(cls, step_time):
        log_dict = {
            "system_status/step_time": step_time,
            "system_status/gpu_memory_reserved": torch.cuda.memory_reserved(0) // 1e9,
            "system_status/gpu_memory_allocated": torch.cuda.memory_allocated(0) // 1e9,
        }
        try:
            log_dict["system_status/cpu_reserved_percent"] = psutil.cpu_percent()
        except OSError as exc:
            cls._warn_system_metrics_unavailable(exc)
        try:
            log_dict["system_status/ram_reserved_percent"] = psutil.virtual_memory().percent
        except OSError as exc:
            cls._warn_system_metrics_unavailable(exc)
        return log_dict
