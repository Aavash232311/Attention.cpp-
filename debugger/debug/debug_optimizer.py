import warnings
import json
# Read the released hyperparamaters .json file from C++ script.
warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")



class AdamW:

    def __init__(self, beta_1, beta_2, epsilon, lr, wd):
        ...


class DebugOptimizer:

    def __init__(self):
        self.path ="./src/cache/optimizer_config.json"

        self.params = self._read_hyperparamaters()
        self.beta_1 = self.params["beta_1"]
        self.beta_2 = self.params["beta_2"]
        self.epsilon = self.params["epsilon"]
        self.weight_decay = self.params["wd"]
        self.learning_rate = self.params["lr"]

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




