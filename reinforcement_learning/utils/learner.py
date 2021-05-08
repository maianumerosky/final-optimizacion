import torch


class Learner:
    def __init__(self, policy, batch_size):
        self.policy = policy
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(policy.model.parameters(), lr=1e-2)

    def update(self, dataset):
        self.policy.model.train()
        mini_batch = dataset.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        # Convert types
        states = torch.stack(list(map(torch.Tensor, states)))
        next_states = torch.stack(list(map(torch.Tensor, next_states)))
        rewards = torch.Tensor(rewards)
        actions = torch.LongTensor(actions)
        dones = torch.Tensor(dones)

        # Compute loss
        values = self.policy.model(states)
        qs = values.gather(dim=1, index=actions.view(-1, 1)).view(-1)

        next_values = self.policy.model(next_states)
        targets = rewards + next_values.max(dim=1).values * (1. - dones).view(-1)

        loss = ((qs - targets.detach()) ** 2).mean()

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
