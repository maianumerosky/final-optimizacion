import torch


# class Policy:
#     def __init__(self, state_dimension, num_actions):
#         self.model = torch.nn.Sequential(
#             torch.nn.Linear(state_dimension, 50),
#             torch.nn.ReLU(),
#             torch.nn.Linear(50, num_actions)
#         )
#
#     def get_distribution(self, state):
#         state = torch.Tensor(state)
#         values = self.model(state)
#         distribution = torch.softmax(values, dim=0)
#         return distribution.data.cpu().numpy()


class Policy:
    def __init__(self, state_dimension, num_actions):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dimension, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, num_actions)
        )

    def get_distribution(self, state):
        with torch.no_grad():
            self.model.eval()
            state = torch.Tensor(state)
            values = self.model(state)
            distribution = torch.softmax(values, dim=0)
        return distribution.data.cpu().numpy()
