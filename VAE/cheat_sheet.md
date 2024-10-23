<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# **Variational Autoencoder (VAE) Cheat Sheet**

---

### **Key Concepts**

1. **Encoder**: Maps input data \( x \) to a distribution over the latent variable \( z \). The encoder produces:
   - **\( \mu(x) \)**: Mean of the latent variable \( z \).
   - **\( \log(\sigma^2(x)) \)**: Log-variance of \( z \).
  
2. **Decoder**: Maps the latent variable \( z \) back to the input space to reconstruct \( x \).

3. **Reparameterization Trick**: Allows differentiable sampling from the latent space:
   \[
   z = \mu(x) + \sigma(x) \cdot \epsilon, \quad \epsilon \sim N(0, I)
   \]
   - Ensures backpropagation through the sampling process by transforming random sampling into a differentiable operation.

4. **KL Divergence**: Measures how close the learned distribution \( q(z|x) \) is to the prior \( p(z) \) (typically a standard normal distribution):
   \[
   D_{\text{KL}}(q(z|x) || p(z)) = \frac{1}{2} \sum_j \left( 1 + \log(\sigma_j^2) - \sigma_j^2 - \mu_j^2 \right)
   \]

5. **Reconstruction Loss**: Measures how well the decoder can reconstruct the original input \( x \). Typically binary cross-entropy for image data:
   \[
   \text{Reconstruction Loss} = \sum_i \text{BCE}(x_i, \hat{x}_i)
   \]

6. **Evidence Lower Bound (ELBO)**: The objective function maximized in VAEs. It is the sum of:
   - **Reconstruction term**: Measures how well the decoder reconstructs the input.
   - **KL divergence term**: Regularizes the latent space to be close to the prior distribution.
   
   The ELBO is a **lower bound** on the marginal likelihood of the data:
   \[
   \log p(x) \geq \mathcal{L}(x) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) || p(z))
   \]

---

### **Detailed Derivations**

---

### **1. ELBO Derivation**

The **Evidence Lower Bound (ELBO)** is derived by introducing an approximate posterior \( q(z|x) \) and applying **Jensen’s Inequality**.

#### Step-by-Step Derivation

We want to maximize the **marginal likelihood** of the data \( x \):

\[
p(x) = \int p(x, z) \, dz = \int p(x|z) p(z) \, dz
\]

However, this integral is often intractable, so we introduce an approximate posterior \( q(z|x) \), which gives:

\[
p(x) = \int q(z|x) \frac{p(x, z)}{q(z|x)} \, dz
\]

Now, applying **Jensen’s Inequality** to the logarithm of the marginal likelihood:

\[
\log p(x) = \log \int q(z|x) \frac{p(x, z)}{q(z|x)} \, dz \geq \int q(z|x) \log \frac{p(x, z)}{q(z|x)} \, dz
\]

The inequality gives the **ELBO**:

\[
\log p(x) \geq \mathbb{E}_{q(z|x)} \left[ \log \frac{p(x, z)}{q(z|x)} \right]
\]

Breaking down the expression:

\[
\mathcal{L}(x) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) || p(z))
\]

- **First term** \( \mathbb{E}_{q(z|x)}[\log p(x|z)] \): Measures how well the decoder reconstructs the input.
- **Second term** \( D_{\text{KL}}(q(z|x) || p(z)) \): Regularizes the approximate posterior to stay close to the prior.

Thus, the **ELBO** is:

\[
\mathcal{L}(x) = \text{Reconstruction Loss} - \text{KL Divergence}
\]

---

### **2. KL Divergence Derivation (Detailed)**

The **KL divergence** between two Gaussian distributions \( q(z|x) = N(\mu_q, \Sigma_q) \) and \( p(z) = N(0, I) \) is a critical part of the VAE loss function.

#### General KL Divergence Formula

The KL divergence between two distributions \( q(z) \) and \( p(z) \) is:

\[
D_{\text{KL}}(q(z) || p(z)) = \int q(z) \log \frac{q(z)}{p(z)} \, dz
\]

For two multivariate Gaussian distributions:
- \( q(z|x) = N(\mu_q, \Sigma_q) \)
- \( p(z) = N(0, I) \)

The KL divergence between these two Gaussians has a **closed-form solution**.

#### Step-by-Step KL Divergence Derivation

The probability density function for a Gaussian \( N(\mu, \Sigma) \) is:

\[
p(z) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(-\frac{1}{2} (z - \mu)^T \Sigma^{-1} (z - \mu)\right)
\]

Substitute the densities of \( q(z|x) \) and \( p(z) \) into the KL divergence formula:

\[
D_{\text{KL}}(q(z|x) || p(z)) = \mathbb{E}_{q(z|x)} \left[ \log \frac{q(z|x)}{p(z)} \right]
\]

Simplifying:

\[
D_{\text{KL}}(q(z|x) || p(z)) = \frac{1}{2} \left( \text{tr}(\Sigma_q) + \mu_q^T \mu_q - d - \log |\Sigma_q| \right)
\]

In the case where \( \Sigma_q \) is diagonal, i.e., \( \Sigma_q = \text{diag}(\sigma_1^2, \dots, \sigma_d^2) \), the trace and determinant simplify:

\[
D_{\text{KL}}(q(z|x) || p(z)) = \frac{1}{2} \sum_j \left( 1 + \log(\sigma_j^2) - \sigma_j^2 - \mu_j^2 \right)
\]

This is the **closed-form solution** for the KL divergence used in VAEs.

---

### **3. Reparameterization Trick Derivation**

#### Why Do We Need the Reparameterization Trick?

Sampling from \( q(z|x) \) is a **stochastic** process and breaks backpropagation, which requires deterministic operations to compute gradients.

The **reparameterization trick** allows us to sample \( z \) in a way that is differentiable:

\[
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim N(0, I)
\]

Where \( \epsilon \) is random noise sampled from a standard normal distribution \( N(0, I) \), and \( \mu \) and \( \sigma \) are the outputs of the encoder.

#### Derivation

Consider the latent variable \( z \) sampled from \( q(z|x) = N(\mu(x), \Sigma(x)) \). To ensure the sampling is differentiable, we use the reparameterization trick:

1. Compute \( \mu(x) \) and \( \log(\sigma^2(x)) \) from the encoder.
2. Sample \( \epsilon \sim N(0, I) \).
3. Compute \( z \) as:
   \[
   z = \mu(x) + \sigma(x) \cdot \epsilon
   \]

Where \( \sigma(x) = \exp(0.5 \cdot \log(\sigma^2(x))) \).

This formulation ensures that \( z \) is differentiable with respect to \( \mu(x) \) and \( \sigma(x) \), allowing for gradient-based optimization.

---

### **VAE Loss Function**

The loss function for VAEs consists of two parts:
1. **Reconstruction Loss**:
   \[
   \text{Reconstruction Loss} = -\mathbb{E}_{q(z|x)}[\log p(x|z)]
   \]
   For binary data (e.g., images), use binary cross-entropy (BCE).

2. **KL Divergence**:
   \[
   D_{\text{KL}}(q(z|x) || p(z)) = \frac{1}{2} \sum_j \left( 1 + \log(\sigma_j^2) - \sigma_j^2 - \mu

_j^2 \right)
   \]

3. **Total Loss**:
   \[
   \text{VAE Loss} = \text{Reconstruction Loss} + \text{KL Divergence}
   \]

---

### **PyTorch Code**

```python
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_divergence
```

---

### Summary

- **ELBO**: Measures how well the model reconstructs data and how close the latent space is to the prior.
- **KL Divergence**: Ensures the learned latent space stays close to the prior distribution.
- **Reparameterization Trick**: Enables backpropagation through the sampling process.
- **Loss Function**: Combines reconstruction loss and KL divergence to form the VAE objective.

This detailed cheat sheet includes both the **concepts** and **derivations** essential for understanding and implementing VAEs. Let me know if you need any further clarifications!