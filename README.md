# CTM_CFG
Use Continuous Thought Machine to automatically adjust the guidance scale of Classifier Free Guidance. 

https://github.com/Yanfeng-Yang-0316/CTM_CFG/blob/main/jit_ctm.mp4

Classifier-free guidance (CFG) is a well-known technique for injecting and amplifying conditional information when generating images under given conditions. Specifically, CFG replaces the vector field in the above ODE with the interpolation $$v(X_t,t)+w[v(X_t,y,t)-v(X_t,t)]$$, where $$w$$ is a positive guidance strength that controls the strength of the condition $$y$$. The guidance strength $$w$$ in the original CFG is a static constant. A small $$w$$ is often insufficient to make the generated image align well with the condition $$y$$, whereas an excessively large $$w$$ often distorts image details. 

THe motivation of this code is inspired by the recent study [1], which shows that the guidance strength $$w$$ in CFG tends to enhance condition-specific features while suppressing general features. However, their interpretation is derived in a linear diffusion model, where such components can still be analyzed explicitly through the difference of the conditional and unconditional covariance. For realistic nonlinear neural networks, this kind of analytical decomposition is no longer tractable. Therefore, instead of trying to separate condition-specific and general features analytically, I use the internal attention masks of CTM as a data-dependent proxy for where condition-relevant information is located, and then spatially weight the CFG strength accordingly. 

The internal dynamics of Continuous thought machine (CTM) [2] are particularly appealing. They provide an interpretation of which regions of an image the model is actually ‘looking’ at. In other words, the model tends to focus on regions that are more important for the condition $$y$$, while paying less attention to regions that are less relevant to $$y$$. This makes it possible to filter out the general features mentioned in the previous paragraph and instead identify condition-specific features. 

More specifically, let $$τ = 1,2,...,T$$ denote the internal time ticks of CTM. Let $$C=[C_1,...,C_T]$$ denote the certainty scores of CTM at different ticks after softmax normalization, and let $$O=[O_1,..,O_T]$$ denote the corresponding attention masks. I first used tweedie projection ($$E[X_0|X_t,y]$$) to map $$v(X_t,y,t)$$ to the pixel space, input it to CTM, and get $$C$$ and $$O$$. Then, I define the spatial mask as 

$$
M=sum_τ (C_τ*O_τ). 
$$

I then linearly interpolate $$M$$ to the image resolution and normalize it so that its mean is equal to 0. After that, I introduce an additional parameter $$λ$$ to control the scale of the mask and clip $$M$$ to [-1,1]. Finally, I replace the vector field in the original CFG ODE with 

$$
v(X_t,t)+ w*(1+λ*M) \circ [v(X_t,y,t)-v(X_t,t)],
$$ 

where $$\circ$$ denotes elementwise multiplication. The $$(1+λ*M)$$ term can make guidance strength on condition-correlated regions bigger than 1, and make the strength on the uncorrelated regions smaller than 1.


[1] Towards Understanding the Mechanisms of Classifier-Free Guidance, https://arxiv.org/abs/2505.19210.

[2] Continuous Thought Machines, https://openreview.net/forum?id=y0wDflmpLk.
