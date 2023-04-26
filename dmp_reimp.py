import torch
import numpy as np


class DMPs:
    def __init__(self, n_dmps, n_bfs, y0=0, goal=1, w=None, alpha=None, beta=None, dt=0.01):
        """
        Arguments:
            - n_dmps: Int, Number of dynamic motor primitives, equivalent to the number of DoFs we are controlling
            - n_bfs: Int, Number of basis functions per DMP
            - y0: List or Tensor, Initial state of DMPs
            - goal: List or Tensor, Goal state of DMPs (Trainable)
            - w: List or Tensor, Weights that control the amplitude of basis functions (Trainable)
            - alpha: Float, Gain on attractor dynamics for y_dot
            - beta: Float, Gain on attractor dynamics for y
            - dt: Float, Timestep for simulation rollouts
        """
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        
        # Convert initial state into torch array
        if isinstance(y0, (int, float)):
            y0 = torch.ones(self.n_dmps) * y0
        elif isinstance(y0, list):
            y0 = torch.Tensor(y0)
        self.y0 = y0
        
        # Convert goal into torch array
        if isinstance(goal, (int, float)):
            goal = torch.ones(self.n_dmps) * goal
        elif isinstance(goal, list):
            goal = torch.Tensor(goal)
        self.goal = goal

        # Initialize weights
        if w is None:
            w = torch.zeros(self.n_dmps, self.n_bfs)
        self.w = w
        
        # Initialize gains
        self.alpha = alpha if alpha is not None else torch.ones(n_dmps)*25.0
        self.beta = beta if beta is not None else self.alpha/4.0

        self.dt = dt

        # Set up the canonical system
        self.cs = CanonicalSystem(dt=self.dt)
        self.timesteps = int(self.cs.run_time / self.dt)

        # Initialize state
        self.y = None
        self.

        def reset_state(self):
            """Reset the system state"""
            self.y = self.y0.clone()
            self.dy = torch.zeros(self.n_dmps)
            self.ddy = torch.zeros(self.n_dmps)
            self.cs.reset_state()

        def gen_front_term(self, x, dmp_num):
            """Generates the diminishing front term on the forcing term. 

            Arguments:
                - x: Float, The current value of the canonical system
                - dmp_num: Int, the index of the current dmp
            """
            return x * (self.goal[dmp_num] - self.y0[dmp_num])

        def gen_psi(self, x):
            """Generates the basis function activations for a given canonical system rollout
            
            Arguments:
                - x: Float, The current value of the canonical system
            """

            
        def integrate(self, tau=1.0, error=0.0, external_force=None):
            """Run the DMP system for a single timestep.
            
            Arguments:
                - tau: Float, Scales the timestep. Increase it to make the system execute faster
                - error: Float, Optional system feedback
                - external_force: List or Tensor, Optional external force per dynamic motion primitive

            Returns:
                - y: Tensor, State of the system
                - dy: Tensor, Velocity of the system
            """
            # Get error coupling if exists
            error_coupling = 1.0 / (1.0 + error)

            # Run canonical system
            x = self.cs.step(tau=tau, error_coupling=error_coupling)

            # Generate basis function activation
            psi = self.gen_psi(x)

            # Get denominator for RBF, set it to one if it's too small
            sum_psi = torch.sum(psi)
            if sum_psi < 1e-6:
                sum_psi = 1

            for d in range(self.n_dmps):
                # Generate the forcing term
                f = self.gen_front_term(x, d) * (torch.dot(psi, self.w[d]))
                f / sum_psi

                # DMP acceleration
                self.ddy[d] = (
                    alpha[d] * (
                        beta[d] * (self.goal[d] - self.y[d]) - self.dy[d]
                        ) + f
                )
                if external_force is not None:
                    self.ddy[d] += external_force[d]

                # Velocity and state through Euler integration
                self.dy[d] += self.ddy[d] * self.dt * tau * error_coupling
                self.y[d] += self.dy[d] * self.dt * tau * error_coupling

            return self.y, self.dy, self.ddy