# Should converge to ~0.22.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def make_training_data(num_samples, dims, sigma):
    dt = np.asarray(sigma).dtype
    x = np.random.randn(dims, num_samples).astype(dt)
    w = sigma * np.random.randn(1, dims).astype(dt)
    noise = np.random.randn(num_samples).astype(dt)
    y = w.dot(x) + noise
    return y[0], x, w[0]


def make_weights_prior(dims, log_sigma):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([dims], dtype=log_sigma.dtype),
        scale_identity_multiplier=tf.exp(log_sigma))


def make_response_likelihood(w, x):
    if w.shape.ndims == 1:
        y_bar = tf.matmul(w[tf.newaxis], x)[0]
    else:
        y_bar = tf.matmul(w, x)
    return tfd.Normal(loc=y_bar, scale=tf.ones_like(y_bar))  # [n]


# Setup assumptions.
dtype = np.float32
num_samples = 500
dims = 10
tf.compat.v1.random.set_random_seed(10014)
np.random.seed(10014)

weights_prior_true_scale = np.array(0.3, dtype)
y, x, _ = make_training_data(num_samples, dims, weights_prior_true_scale)

log_sigma = tf.Variable(name='log_sigma', initial_value=np.array(0, dtype))

optimizer = tf.optimizers.SGD(learning_rate=0.01)


@tf.function
def mcem_iter(weights_chain_start, step_size):
    with tf.GradientTape() as tape:
        tape.watch(log_sigma)
        prior = make_weights_prior(dims, log_sigma)

        def unnormalized_posterior_log_prob(w):
            likelihood = make_response_likelihood(w, x)
            return (prior.log_prob(w) +
                    tf.reduce_sum(input_tensor=likelihood.log_prob(y), axis=-1)
                    )  # [m]

        def trace_fn(_, pkr):
            return (pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.accepted_results.target_log_prob,
                    pkr.inner_results.accepted_results.step_size)

        num_results = 2
        weights, (
            log_accept_ratio, target_log_prob,
            step_size) = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=0,
                current_state=weights_chain_start,
                kernel=tfp.mcmc.SimpleStepSizeAdaptation(
                    tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=unnormalized_posterior_log_prob,
                        num_leapfrog_steps=2,
                        step_size=step_size,
                        state_gradients_are_stopped=True,
                    ),
                    # Adapt for the entirety of the trajectory.
                    num_adaptation_steps=2),
                trace_fn=trace_fn,
                parallel_iterations=1)

        # We do an optimization step to propagate `log_sigma` after two HMC
        # steps to propagate `weights`.
        loss = -tf.reduce_mean(input_tensor=target_log_prob)

    avg_acceptance_ratio = tf.reduce_mean(
        input_tensor=tf.exp(tf.minimum(log_accept_ratio, 0.)))

    optimizer.apply_gradients([[tape.gradient(loss, log_sigma), log_sigma]])

    weights_prior_estimated_scale = tf.exp(log_sigma)
    return (weights_prior_estimated_scale, weights[-1], loss, step_size[-1],
            avg_acceptance_ratio)


num_iters = int(40)

weights_prior_estimated_scale_ = np.zeros(num_iters, dtype)
weights_ = np.zeros([num_iters + 1, dims], dtype)
loss_ = np.zeros([num_iters], dtype)
weights_[0] = np.random.randn(dims).astype(dtype)
step_size_ = 0.03

for iter_ in range(num_iters):
    [
        weights_prior_estimated_scale_[iter_],
        weights_[iter_ + 1],
        loss_[iter_],
        step_size_,
        avg_acceptance_ratio_,
    ] = mcem_iter(weights_[iter_], step_size_)
    tf.compat.v1.logging.vlog(
        1, ('iter:{:>2}  loss:{: 9.3f}  scale:{:.3f}  '
            'step_size:{:.4f}  avg_acceptance_ratio:{:.4f}').format(
                iter_, loss_[iter_], weights_prior_estimated_scale_[iter_],
                step_size_, avg_acceptance_ratio_))

plt.plot(weights_prior_estimated_scale_)
plt.ylabel('weights_prior_estimated_scale')
plt.xlabel('iteration')
plt.show()
