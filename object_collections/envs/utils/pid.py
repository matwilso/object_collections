
class PID(object):
    """Simple PD loop implementation for moving the paddle"""
    def __init__(self, P, D, curr_fn, goal=None):
        self.P = P
        self.D = D
        self.curr_fn = curr_fn
        self.goal = goal
        self.prev_dp = 0.0  # 0 may not be the best option as it may be jerkier on the startup

    def reset_goal(self, goal):
        self.goal = goal
        curr = self.curr_fn()
        self.prev_dp =  self.goal - curr
        #self.prev_dp = 0.0

    def step(self):
        curr = self.curr_fn()
        dp = self.goal - curr

        ddp = dp - self.prev_dp
        self.prev_dp = dp

        return self.P * dp, self.D * ddp