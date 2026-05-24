# CTM_CFG
Use Continuous Thought Machine to automatically adjust the guidance scale of Classifier Free Guidance. 

https://github.com/user-attachments/assets/f4782f12-b876-4191-bd0b-73ede19d8064 

# Introduction
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


# Usage

The generative model is Just image Transformers (JIT) [3]. Please download their github code and model, and CTM's code, then run the notebook file.

# Requirements

Please install CTM's env or JIT's env. Their env are highly interoperable.

Remind: Please use Reflect padding in the ResNet (many thanks to a very clever guy). Specifically, change this function in https://github.com/SakanaAI/continuous-thought-machines/blob/main/models/resnet.py#L352:
```bash
def prepare_resnet_backbone(backbone_type):
      
    resnet_family = resnet18 # Default
    if '34' in backbone_type: resnet_family = resnet34
    if '50' in backbone_type: resnet_family = resnet50
    if '101' in backbone_type: resnet_family = resnet101
    if '152' in backbone_type: resnet_family = resnet152

    # Determine which ResNet blocks to keep
    block_num_str = backbone_type.split('-')[-1]
    hyper_blocks_to_keep = list(range(1, int(block_num_str) + 1)) if block_num_str.isdigit() else [1, 2, 3, 4]

    backbone = resnet_family(
        3,
        hyper_blocks_to_keep,
        stride=2,
        pretrained=False,
        progress=True,
        device="cpu",
        do_initial_max_pool=True,
    )
    for m in backbone.modules():
        if isinstance(m, nn.Conv2d) and m.padding != (0, 0):
            m.padding_mode = "reflect"

    return backbone
```

[1] Towards Understanding the Mechanisms of Classifier-Free Guidance, https://arxiv.org/abs/2505.19210.

[2] Continuous Thought Machines, https://openreview.net/forum?id=y0wDflmpLk.

[3] Back to Basics: Let Denoising Generative Models Denoise, https://arxiv.org/abs/2511.13720.
