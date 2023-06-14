## VAE Basics

Variational Autoencoders are generative latent-variable models that are popularly used for unsupervised learning and are aimed at maximizing the log-likelihood of the data, that is, maximizing $\sum\limits_{i=1}^N \log p(x_i; \theta)$ where $N$ is the number of data samples available. The generative story is as follows:

- $z \sim \mathcal{N}(0, I)$

- $x | z \sim \mathcal{N}(\mu_\theta(z), \Sigma_\theta(z))$

Given $\mu_\theta(\cdot)$ and $\Sigma_\theta(\cdot)$ are parameterized as arbitrary Neural Networks, one cannot obtain the log-likelihood $\log \mathbb{E}_{z}[p(x | z, \theta)]$ in closed form and hence has to rely on variational assumptions for optimization.

One way of optimizing for log-likelihood is to use the variational distribution $q_\phi(z | x)$, which with a little bit of algebra leads to the ELBO, which is:

![equation](https://latex.codecogs.com/svg.latex?ELBO%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Cleft%28%20%5Cmathbb%7BE%7D_%7Bz%5Csim%20q_%5Cphi%28z%7Cx_i%29%7D%20%5B%5Clog%20p_%5Ctheta%28x_i%20%7C%20z%29%5D%20%2B%20%5Cmathbb%7BKL%7D%5Bq_%5Cphi%28z%7Cx_i%29%20%7C%7C%20%5Cmathcal%7BN%7D%280%2C%20I%29%5D%20%5Cright%29)

This is the objective that we use for optimizing VAEs, where different flavours of VAE can be obtained by changing either the approximate posterior $q_\phi$, the conditional likelihood distribution $p_\theta$ or even the standard normal prior.

In this repo we have implemented a simple version of a VAE, where $q_\phi(z|x)$ will be parameterized as $\mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$ where $\mu(x)$ is a mean vector and $\Sigma(x)$ will be a **diagonal covariance matrix**, that is, it will only have non-zero entries on the diagonal.

The likelihood $p_\theta(x|z)$ will also be modeled as a Gaussian Distribution $\mathcal{N}(\mu_\theta(z), I)$ where we parameterize the mean with another neural network but for simplicity, consider the identity covariance matrix.

# Diagonal Gaussian Distribution

- Sampling: Provide the methodology of computing a **reparamterized** sample from the given distribution.
- KL Divergence: Compute and return the KL divergence of the distribution with the standard normal, that is, $\mathbb{KL}[\mathcal{N}(\mu, \Sigma) || \mathcal{N}(0, I)]$ where $\Sigma$ is a diagonal covariance matrix.
- Negative Log Likelihood: Given some data $x$, returns the log likelihood under the current gaussian, that is, $\log \mathcal{N}(x | \mu, \Sigma)$
- Mode: Returns the mode of the distribution 

# VAE Model

The Variational Autoencoder (VAE) model consists of an encoder network that parameterizes the distribution $q_\phi$ as a Diagonal Gaussian Distribution through the (mean, log variance) parameterization and a decoder network that parameterizes the distribution $p_\theta$ as another Diagonal Gaussian Distribution with an identity covariance matrix.

- Encode: The function that takes as input a batched data sample, and returns the approximate posterior distribution $q_\phi$
- Decode: The function that takes as input a batched sample from the latent space, and returns the mode of the distribution $p_\theta$
- Sample: Generates a novel sample by sampling from the prior and then using the mode of the distribution $p_\theta$
- Forward: The main function for training. Given a data sample x, encode it using the encode function, and then obtain a reparameterized sample from it, and finally decode it. Return the mode from the decoded distribution $p_\theta$, as well as the conditional likelihood and KL terms of the loss.
- Log Likelihood: The main function for testing that approximates the log-likelihood of the given data. It is computed using importance sampling as $\log \frac{1}{K} \sum\limits_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k|x)}$ where $z_k \sim q_\phi(z | x)$.
