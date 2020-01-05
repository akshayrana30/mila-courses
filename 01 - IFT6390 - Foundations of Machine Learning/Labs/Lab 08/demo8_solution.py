### We initialize weights and parameters
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# set requires_grad = True to track the gradients
weights = torch.randn(num_features, num_classes, device=device, requires_grad=True)

learning_rate = 2e-5
num_epochs = 500

losses = []

### We now perform Gradient Descent
### On effectue la descente de gradient

for epoch in range(num_epochs):

    # QUESTION : compute the predictions and MSE loss
    # (Hint : torch.mm(x,y) is matrix multiplication in PyTorch)
    preds = torch.mm(x_train, weights)  # predictions
    loss = (preds - y_train).pow(2).mean()  # MSE loss
    losses.append(loss.item())

    ### MANUAL gradient computation
    # QUESTION : complete the gradient computation for MSE
    grad_preds = 2.0 * (preds - y_train)  # gradients wrt predictions
    grad_w = torch.mm(x_train.t(), grad_preds) / (num_train * num_classes)  # gradients wrt weights

    ### AUTO gradients computation
    loss.backward()

    # Let's make sure both computations are equivalent
    if not np.allclose(grad_w.tolist(), weights.grad.tolist()):
        raise BaseException('gradients are different !')

    ### Update the weights without changing the gradients
    with torch.no_grad():
        # access the gradient with weights.grad
        weights -= learning_rate * weights.grad

        # make sure to reset the gradient to 0 for our next calculation
        weights.grad.zero_()