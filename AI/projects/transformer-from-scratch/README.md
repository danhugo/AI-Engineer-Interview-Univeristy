Embedding Block

Mapping token ID to corresponding weight vector in the embedding table. The weights matrix is initialized following normal distribution. With padding_idx, it the embedding at padding_idx will not contribute to gradient descent. Therefore it is ussually initialized as 0 and is clean out after gradient. A good trick to clean out gradient at position `padding_idx` is to use register_hook.

what regiester_hook do is: it change the the gradient value before adding back to the weight

gradient compute -> register_hook: clean out gradient at index padding_idx -> weight += gradient

Positional Encoding block

PE(pos, 2i) = sin(pos / 10000 ^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000 ^ (2i/d_model))

in pytorch: compute 10000 ^ (2i/d_model) directly might lead to precision loss, numerical instability or non hardware compute utilization.

there for it is insead compute by e^(ln10000 * 2i / d_model)

Why 10000 is the scale number in positional encoding for transformer. It is not a magic number.

It is choosen to make sure pe(pos) is not repeated within the window length of N tokens.