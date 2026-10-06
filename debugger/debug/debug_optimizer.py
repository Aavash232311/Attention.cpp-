import warnings
import json
# Read the released hyperparamaters .json file from C++ script.
warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")


import torch
from debug.debug_flash_attention import DebugFlashAttention
from binary_reader.autograd_binary_reader import load_optimized_grad
from debug.static import RESET, RED, GREEN


class DebugOptimizer(DebugFlashAttention):

    def __init__(self, batch_size, seq_len, vocab_size, d_model, num_heads, head_dim, dl_dw: torch.Tensor):
        self.path ="./src/cache/optimizer_config.json"

        self.params = self._read_hyperparamaters()
        self.beta_1 = self.params["beta_1"]
        self.beta_2 = self.params["beta_2"]
        self.epsilon = self.params["epsilon"]
        self.weight_decay = self.params["wd"]
        self.lr = self.params["lr"]
        self.t = self.params["t"]

        super().__init__(batch_size, seq_len, vocab_size, d_model, num_heads, head_dim, dl_dw)

        (self.d_weight_q_optimal) = load_optimized_grad(batch_size, seq_len, vocab_size, d_model, num_heads, head_dim)


        # print(self.params, self.beta_1, self.beta_2, self.epsilon, self.weight_decay, self.learning_rate)

    def _read_hyperparamaters(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                return data

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Critical Error: {e}")
            print("Exiting program. Couldn't read the hyperparamaters, try build again with the debugger flag.")
            sys.exit(1)

    def _optimize(self):
        param = torch.nn.Parameter(self.wq.detach().clone())

        optimizer = torch.optim.AdamW(
            [param],
            lr=self.lr,
            betas=(self.beta_1, self.beta_2),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )

        for _ in range(self.t):
            param.grad = self.d_weight_q
            optimizer.step()

        return param.detach()


    def optimizer_health(self):

        # check this one in loose precision
        torch_wq_optimal = self._optimize()

        check_optimizer = torch.allclose(
            self.d_weight_q_optimal,
            torch_wq_optimal,
            atol=1e-4,
            rtol=1e-4,
        )


        print("\n")
        print("*" * 60)
        print("AdamW C++ kernel status")
        print("*" * 60)
        print("\n")

        # Here arg of mine is if it works for the one kernel then it works for every kernel
        # until and unless you passed something.
        # we will make this as arb
        if not check_optimizer:
            print(f"AadamW C++ kernel status: {RED} {check_optimizer} {RESET}")
        else:
            print(f"AdamW C++ kernel status: {GREEN} {check_optimizer} {RESET}")

        print("Optimized from kernel")
        print(self.d_weight_q_optimal)

        print("Optimized from torch")
        print(torch_wq_optimal)
