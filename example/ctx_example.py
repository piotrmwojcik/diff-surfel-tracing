import torch


class Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        a_weights = x + 1  # our weights
        result = x * a_weights  # simulate rasterization

        return result, a_weights  # Returning both

    @staticmethod
    def backward(ctx, grad_result, grad_a_weights):  # exactly like in surfel tracer
        (x,) = ctx.saved_tensors  # ctx skips a_weights

        # compute gradients for x. FOR a_weights NO GRADIENT COMPUTATION HERE
        grad_x = grad_result * 2 * x

        return grad_x


# Use the custom function
x = torch.tensor(3.0, requires_grad=True)
light = torch.tensor(7.0, requires_grad=True)

# Apply the function
result, a_weights = Rasterize.apply(x)

# Only use `result` in the loss – like how `a_weights` is unused in backward
loss = (light ** 2) * a_weights * result  # Equivalent to: light * transmittance* raster

loss.backward()

# Print gradients
print(f'Gradient of x: {x.grad.item()}')
print(f'Gradient of light: {light.grad.item()}')
print(f'Gradient of a: {a_weights.grad}')  # grad here is NOne