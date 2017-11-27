class RandomAction():

    def __init__(self, env):
        self.env = env
        self.replay_memory = None

    def play(self):
        self.replay_memory = []

        state = self.env.reset()
        is_reset = False
        for t in range(100000):

            if is_reset:
                state = self.env.reset()

            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.replay_memory.append([state, action, next_state])

            if done:
                is_reset = True

        return self.replay_memory
