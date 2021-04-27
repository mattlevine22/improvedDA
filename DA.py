import os
import numpy as np
import pickle
from pydoc import locate

from computation_utils import computeErrors
from plotting_utils import *
from odelibrary import my_solve_ivp

import pdb

def get_Psi(dt, rhs, integrator, t0=0):
	t_span = [t0, t0+dt]
	t_eval = np.array([t0+dt])
	settings = {}
	settings['dt'] = dt
	settings['method'] = integrator
	return lambda ic0: my_solve_ivp(ic=ic0, f_rhs=lambda t, y: rhs(y, t), t_eval=t_eval, t_span=t_span, settings=settings)

def generate_data(dt, t_eval, ode, t0=0):
	ic = ode.get_inits()
	t_span = [t_eval[0], t_eval[-1]]
	settings = {}
	settings['method'] = 'RK45'
	return my_solve_ivp(ic=ic, f_rhs=lambda t, y: ode.rhs(y, t), t_eval=t_eval, t_span=t_span, settings=settings)

class VAR3D(object):
	def __init__(self, H, output_dir='default_output_3DVAR',
				K=None, dt=0.01, T=100, dynamics_rhs='L63',
				integrator='RK45', t0=0, x_ic=None, obs_noise_sd=0.1,
				lr=0.005):

		# create output directory
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self.H = H # linear observation operator
		self.t0 = t0
		self.t_pred = t0
		self.t_assim = t0
		self.dt = dt
		self.T = T
		self.obs_noise_sd = obs_noise_sd
		self.lr = lr

		# set up dynamics
		ODE = locate('odelibrary.{}'.format(dynamics_rhs))
		self.ode = ODE()
		self.Psi = get_Psi(dt=self.dt, rhs=self.ode.rhs, integrator=integrator)

		# set up true data
		self.reset_data()

		# set up observation data
		dim_x = self.x_true.shape[1]
		dim_y = self.H.shape[0]
		obs_noise_mean = np.zeros(dim_y)
		obs_noise_cov = (obs_noise_sd**2) * np.eye(dim_y)
		self.y_obs = (self.H @ self.x_true.T).T + np.random.multivariate_normal(mean=obs_noise_mean, cov=obs_noise_cov, size=self.N)

		# set up DA arrays
		self.x_pred = np.zeros_like(self.x_true)
		self.y_pred = np.zeros_like(self.y_obs)
		self.x_assim = np.zeros_like(self.x_true)

		# set up useful DA matrices
		self.Ix = np.eye(dim_x)

		# choose ic for DA
		if x_ic is None:
			x_ic = self.ode.get_inits()
		self.x_assim[0] = x_ic
		self.x_pred[0] = x_ic
		self.y_pred[0] = self.H @ x_ic

		# set default gain to 0
		if K is None:
			K =  np.zeros(dim_x, dim_y) # linear gain
		self.K = K

	def reset_data(self):
		self.times = np.arange(self.t0, self.T+self.dt, self.dt)
		self.N = len(self.times)
		self.x_true = generate_data(dt=self.dt, t_eval=self.times, ode=self.ode, t0=self.t0)

	def predict(self, ic):
		return self.Psi(ic)

	def update(self, x_pred, y_obs):
		return (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)

	def test_filter(self):
		# DA @ c=0, t=0 has been initialized already
		for c in range(1, self.N):
			# predict
			self.t_pred += self.dt
			self.x_pred[c] = self.predict(ic=self.x_assim[c-1])
			self.y_pred[c] = self.H @ self.x_pred[c]

			# assimilate
			self.t_assim += self.dt
			self.x_assim[c] = self.update(x_pred=self.x_pred[c], y_obs=self.y_obs[c])

		# compute evaluation statistics
		self.eval_dict = computeErrors(target=self.x_true, prediction=self.x_assim, dt=self.dt, thresh=self.obs_noise_sd)

		# plot assimilation errors
		fig_path = os.path.join(self.output_dir, 'assimilation_errors_all')
		plot_assimilation_errors(times=self.times, errors=self.eval_dict['mse'], eps=self.obs_noise_sd, fig_path=fig_path)


	def train_filter(self):
		self.K_vec = np.zeros( (self.N, self.K.shape[0], self.K.shape[1]) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)
		self.K_vec[0] = np.copy(self.K)
		self.K_vec_runningmean[0] = np.copy(self.K)
		self.loss = np.zeros(self.N-1)

		# DA @ c=0, t=0 has been initialized already
		for c in range(1, self.N):
			# predict
			self.t_pred += self.dt
			self.x_pred[c] = self.predict(ic=self.x_assim[c-1])
			self.y_pred[c] = self.H @ self.x_pred[c]

			# compute loss
			m_c = self.update(x_pred=self.x_pred[c], y_obs=self.y_obs[c])
			L_c = np.sum( (m_c - self.x_true[c])**2 ) #|| ||^2
			self.loss[c-1] = L_c

			# compute gradient of loss wrt K
			pred_err = (self.y_obs[c,None].T - self.y_pred[c,None].T).T
			grad_loss = 2*( self.K*pred_err + self.x_pred[c,None].T - self.x_true[c,None].T) @ pred_err.T

			# update K
			self.K -= self.lr * grad_loss
			self.K_vec[c] = np.copy(self.K)
			self.K_vec_runningmean[c] = np.mean(self.K_vec[:c+1], axis=0)

			# assimilate
			self.t_assim += self.dt
			self.x_assim[c] = self.update(x_pred=self.x_pred[c], y_obs=self.y_obs[c])

		# plot K convergence
		fig_path = os.path.join(self.output_dir, 'K_learning_sequence')
		plot_K_learning(times=self.times, K_vec=self.K_vec, fig_path=fig_path)

		fig_path = os.path.join(self.output_dir, 'K_learning_runningMean')
		plot_K_learning(times=self.times, K_vec=self.K_vec_runningmean, fig_path=fig_path)

		# plot learning error
		fig_path = os.path.join(self.output_dir, 'loss_sequence')
		plot_loss(times=self.times[:-1], loss=self.loss, fig_path=fig_path)


class ENKF(object):
	def __init__(self, H, output_dir='default_output_EnKF',
				N_particles=100,
				obs_noise_sd=0.1,
				state_noise_sd=0,
				x_ic_mean=None,
				x_ic_sd=10,
				s_perturb_obs=True,
				dt=0.01, T=100, dynamics_rhs='L63',
				integrator='RK45', t0=0):

		# create output directory
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self.N_particles = N_particles
		self.H = H # linear observation operator
		self.t0 = t0
		self.t_pred = t0
		self.t_assim = t0
		self.dt = dt
		self.T = T
		self.obs_noise_sd = obs_noise_sd
		self.state_noise_sd = state_noise_sd
		self.s_perturb_obs = s_perturb_obs

		# set up dynamics
		ODE = locate('odelibrary.{}'.format(dynamics_rhs))
		self.ode = ODE()
		self.Psi = get_Psi(dt=self.dt, rhs=self.ode.rhs, integrator=integrator)

		# set up true data
		self.reset_data()

		# set up observation data
		dim_x = self.x_true.shape[1]
		dim_y = self.H.shape[0]
		self.obs_noise_mean = np.zeros(dim_y)
		self.Gamma = (obs_noise_sd**2) * np.eye(dim_y) # obs_noise_cov
		self.y_obs = (self.H @ self.x_true.T).T + np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma, size=self.N)

		# set up DA arrays
		self.x_pred_mean = np.zeros_like(self.x_true)
		self.y_pred_mean = np.zeros_like(self.y_obs)
		self.x_assim_mean = np.zeros_like(self.x_true)

		self.x_pred_particles = np.zeros( (self.N, self.N_particles, dim_x) )
		self.y_pred_particles = np.zeros( (self.N, self.N_particles, dim_y) )
		self.x_assim_particles = np.zeros( (self.N, self.N_particles, dim_x) )
		self.x_assim_error_particles = np.zeros( (self.N, self.N_particles, dim_x) )

		self.x_pred_cov = np.zeros((self.N, dim_x, dim_x))

		# set up useful DA matrices
		self.Ix = np.eye(dim_x)
		self.K_vec = np.zeros( (self.N, dim_x, dim_y) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)

		# choose ic for DA
		x_ic_cov = (x_ic_sd**2) * np.eye(dim_x)
		if x_ic_mean is None:
			x_ic_mean = np.zeros(dim_x)

		x0 = np.random.multivariate_normal(mean=x_ic_mean, cov=x_ic_cov, size=self.N_particles)
		self.x_assim_particles[0] = np.copy(x0)
		self.x_pred_particles[0] = np.copy(x0)

		self.x_pred_mean[0] = np.mean(x0, axis=0)
		self.y_pred_mean[0] = self.H @ self.x_pred_mean[0]


	def reset_data(self):
		self.times = np.arange(self.t0, self.T+self.dt, self.dt)
		self.N = len(self.times)
		if self.state_noise_sd > 0:
			raise ValueError('Data generation not yet set up for stochastic dynamics')
		else:
			self.x_true = generate_data(dt=self.dt, t_eval=self.times, ode=self.ode, t0=self.t0)

	def predict(self, ic):
		return self.Psi(ic)

	def update(self, x_pred, y_obs):
		return (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)

	def test_filter(self):
		# DA @ c=0, t=0 has been initialized already
		for c in range(1, self.N):
			## predict
			self.t_pred += self.dt
			# compute and store ensemble forecasts
			for n in range(self.N_particles):
				self.x_pred_particles[c,n] = self.predict(ic=self.x_assim_particles[c-1,n])
				self.y_pred_particles[c,n] = self.H @ self.x_pred_particles[c,n]
			# compute and store ensemble means
			self.x_pred_mean[c] = np.mean(self.x_pred_particles[c], axis=0)
			self.y_pred_mean[c] = self.H @ self.x_pred_mean[c]

			# compute and store ensemble covariance
			C_hat = np.cov(self.x_pred_particles[c], rowvar=False)
			self.x_pred_cov[c] = C_hat

			## compute gains for analysis step
			S = self.H @ C_hat @ self.H.T + self.Gamma
			self.K = C_hat @ self.H.T @ np.linalg.inv(S)
			self.K_vec[c] = np.copy(self.K)
			self.K_vec_runningmean[c] = np.mean(self.K_vec[:c], axis=0)


			## assimilate
			self.t_assim += self.dt
			for n in range(self.N_particles):
				# optionally perturb the observation
				y_obs_n = self.y_obs[c] + self.s_perturb_obs * np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma)

				# update particle
				self.x_assim_particles[c,n] = self.update(x_pred=self.x_pred_particles[c,n], y_obs=y_obs_n)

			# compute and store ensemble means
			self.x_assim_mean[c] = np.mean(self.x_assim_particles[c], axis=0)

		# compute evaluation statistics
		self.eval_dict = computeErrors(target=self.x_true, prediction=self.x_assim_mean, dt=self.dt, thresh=self.obs_noise_sd)

		# plot assimilation errors
		fig_path = os.path.join(self.output_dir, 'assimilation_errors_all')
		plot_assimilation_errors(times=self.times, errors=self.eval_dict['mse'], eps=self.obs_noise_sd, fig_path=fig_path)

		# plot K convergence
		fig_path = os.path.join(self.output_dir, 'K_sequence')
		plot_K_learning(times=self.times, K_vec=self.K_vec, fig_path=fig_path)

		fig_path = os.path.join(self.output_dir, 'K_runningMean')
		plot_K_learning(times=self.times, K_vec=self.K_vec_runningmean, fig_path=fig_path)


class KF(object):
	def __init__(self, H, A,
				output_dir='default_output_KF',
				obs_noise_sd=0.1,
				state_noise_sd=0,
				x_ic_mean=None,
				x_ic_sd=10,
				dt=0.01, T=100, dynamics_rhs='L63',
				integrator='RK45', t0=0):

		# x_k+1 = Ax + \xi,    \xi \sim N(0, \Sigma)
		# y_k+1 = Hx + \eta,    \eta \sim N(0, \Gamma)

		# https://python-control.readthedocs.io/en/0.9.0/generated/control.lqe.html#control.lqe
		# G = I
		# C = H
		# B = D = 0
		# QN -> Sigma
		# RN -> Gamma
		# x_e = Ax_e + Bu + L(y - Cx_e - Du)

		# create output directory
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self.A = A
		self.H = H # linear observation operator
		self.t0 = t0
		self.t_pred = t0
		self.t_assim = t0
		self.dt = dt
		self.T = T
		self.obs_noise_sd = obs_noise_sd
		self.state_noise_sd = state_noise_sd

		# set up true data
		self.reset_data()

		# set up observation data
		dim_x = self.x_true.shape[1]
		dim_y = self.H.shape[0]
		self.obs_noise_mean = np.zeros(dim_y)
		self.Gamma = (obs_noise_sd**2) * np.eye(dim_y) # obs_noise_cov
		self.y_obs = (self.H @ self.x_true.T).T + np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma, size=self.N)

		# set up DA arrays
		self.x_pred_mean = np.zeros_like(self.x_true)
		self.y_pred_mean = np.zeros_like(self.y_obs)
		self.x_assim_mean = np.zeros_like(self.x_true)

		self.x_pred_cov = np.zeros((self.N, dim_x, dim_x))
		self.x_assim_cov = np.zeros((self.N, dim_x, dim_x))

		# set up useful DA matrices
		self.Ix = np.eye(dim_x)
		self.K_vec = np.zeros( (self.N, dim_x, dim_y) )
		self.K_vec_runningmean = np.zeros_like(self.K_vec)

		# choose ic for DA
		x_ic_cov = (x_ic_sd**2) * np.eye(dim_x)
		if x_ic_mean is None:
			x_ic_mean = np.zeros(dim_x)

		x0 = np.random.multivariate_normal(mean=x_ic_mean, cov=x_ic_cov)
		self.x_pred_mean[0] = x0
		self.y_pred_mean[0] = self.H @ x0

	def Psi(self, x):
		return self.A @ x

	def reset_data(self):
		self.times = np.arange(self.t0, self.T+self.dt, self.dt)
		self.N = len(self.times)
		if self.state_noise_sd > 0:
			raise ValueError('Data generation not yet set up for stochastic dynamics')
		else:
			self.x_true = generate_data(dt=self.dt, t_eval=self.times, ode=self.ode, t0=self.t0)

	def predict(self, ic):
		return self.Psi(ic)

	def update(self, x_pred, y_obs):
		return (self.Ix - self.K @ self.H) @ x_pred + (self.K @ y_obs)

	def test_filter(self):
		# DA @ c=0, t=0 has been initialized already
		for c in range(1, self.N):
			## predict
			self.t_pred += self.dt
			# compute and store ensemble forecasts
			for n in range(self.N_particles):
				self.x_pred_particles[c,n] = self.predict(ic=self.x_assim_particles[c-1,n])
				self.y_pred_particles[c,n] = self.H @ self.x_pred_particles[c,n]
			# compute and store ensemble means
			self.x_pred_mean[c] = np.mean(self.x_pred_particles[c], axis=0)
			self.y_pred_mean[c] = self.H @ self.x_pred_mean[c]

			# compute and store ensemble covariance
			C_hat = np.cov(self.x_pred_particles[c], rowvar=False)
			self.x_pred_cov[c] = C_hat

			## compute gains for analysis step
			S = self.H @ C_hat @ self.H.T + self.Gamma
			self.K = C_hat @ self.H.T @ np.linalg.inv(S)
			self.K_vec[c] = np.copy(self.K)
			self.K_vec_runningmean[c] = np.mean(self.K_vec[:c], axis=0)


			## assimilate
			self.t_assim += self.dt
			for n in range(self.N_particles):
				# optionally perturb the observation
				y_obs_n = self.y_obs[c] + self.s_perturb_obs * np.random.multivariate_normal(mean=self.obs_noise_mean, cov=self.Gamma)

				# update particle
				self.x_assim_particles[c,n] = self.update(x_pred=self.x_pred_particles[c,n], y_obs=y_obs_n)

			# compute and store ensemble means
			self.x_assim_mean[c] = np.mean(self.x_assim_particles[c], axis=0)

		# compute evaluation statistics
		self.eval_dict = computeErrors(target=self.x_true, prediction=self.x_assim_mean, dt=self.dt, thresh=self.obs_noise_sd)

		# plot assimilation errors
		fig_path = os.path.join(self.output_dir, 'assimilation_errors_all')
		plot_assimilation_errors(times=self.times, errors=self.eval_dict['mse'], eps=self.obs_noise_sd, fig_path=fig_path)

		# plot K convergence
		fig_path = os.path.join(self.output_dir, 'K_sequence')
		plot_K_learning(times=self.times, K_vec=self.K_vec, fig_path=fig_path)

		fig_path = os.path.join(self.output_dir, 'K_runningMean')
		plot_K_learning(times=self.times, K_vec=self.K_vec_runningmean, fig_path=fig_path)
