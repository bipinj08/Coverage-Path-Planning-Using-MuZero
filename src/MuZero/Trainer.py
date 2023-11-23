from src.CPP.State import CPPState

class TrainerParams:
    def __init__(self):
        self.load_model = ""
        self.num_steps = 2e6
        self.eval_period = 5

class Trainer:
    def __init__(self, params: TrainerParams, agent):
        self.params = params
        self.agent = agent
        self.prefill_bar = None

class MuzeroTrainer(Trainer):
    def __init(self, params, agent):
        super().__init__(params,agent)


