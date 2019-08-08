import structures.const as const
import random
import torch


class SumTree:
    def __init__(self, capacity):
        # holds at most $capacity transitions
        self.memory = []
        # binary sum tree implemented as array starting from index 1
        # index 0 is empty, next $((capacity - 1)) elements hold sums
        # last $capacity elements hold actual priorities
        self.priorities = np.zeros((2*capacity,))   
        sel.max_prio = 1.0
        self.alpha = const.ALPHA
        self.beta = const.BETA
        self.capacity = capacity     # capacity has to be a power of 2
        self.pos = 0

    #def __init__(self, capacity, prob_alpha=0.6):
    def _update_tree(self, pos):
        """
        Update self.priorities starting from pos index.
        """
        node = self.capacity + pos
        while node > 1:
            parent = node // 2
            self.priorities[parent_node] = self.priorities[2*parent] + self.priorities[2*parent+1]
            node = parent

    def _traverse_down(self, val):
        """
        Traverses sum tree down, strating from the root.
        Returs index from priorities array matching given value.
        """
        index = 1
        while 2*index < len(self.priorities):
            if val > self.priorities[2*index] and self.priorities[2*index+1] > 0:
                val -= self.priorities[2*index]
                index = index*2 + 1
            else:
                index *= 2
        return index
        
    
    def push(self, state, action, reward, next_state, done):
        """
        Insert transition with max priority
        """
        state = torch.cat(tuple(state), dim=1)
        next_state = torch.cat(tuple(next_state), dim=1)
        max_prio = self.max_prio if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.capacity + self.pos] = max_prio
        self._update_tree(sel.pos)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self):
        prio_step = self.priorities[1]/batch_size
        sampled_values = [random.uniform(prio_step * k, prio_step * (k+1))
                            for k in range(self.BATCH_SIZE)]

        indices = np.array([self._traverse_down(val) for val in sampled_values])
        samples = [self.memory[i - self.capacity] for i in indices]

        priorities = self.priorities[indices]
        probs = priorities / self.priorities[1]

        total    = len(self.memory)
        weights  = (total * probs) ** (-self.beta)
        weights /= weights.max()
        
        batch       = zip(*samples)
        states      = torch.cat(batch[0]).cuda()
        actions     = torch.tensor(batch[1]).cuda()
        rewards     = torch.tensor(batch[2]).cuda()
        next_states = torch.cat(batch[3]).cuda()
        dones       = torch.tensor(batch[4]).cuda()
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            if prio > self.max_prio:
                self.max_prio = prio
            self.priorities[idx] = prio
            self._update_tree(idx)

    def __len__(self):
        return len(self.memory)


