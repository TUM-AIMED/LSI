"""
File containing the banana experiments
"""


import torch
import numpy as np

from laplace import Laplace
import utils.geometry as geometry
from torch import nn
from utils.manifold import cross_entropy_manifold
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from copy import deepcopy


def riemann(device, 
         seed, 
         x_train, 
         y_train, 
         trained_model, 
         initial_model, 
         subset_of_weights, 
         optimize_prior, 
         hessian_structure,
         weight_decay,
         n_last_layer_weights,
         n_posterior_samples,
         batch_data):
    

    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed: ", seed)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=265, shuffle=True)


    model = trained_model

    # at the end of the training I can get the map solution
    map_solution = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()

    torch.nn.utils.vector_to_parameters(map_solution, model.parameters())


    print("Fitting Laplace")
    la = Laplace(
        model,
        "classification",
        subset_of_weights=subset_of_weights,
        hessian_structure=hessian_structure,
        prior_precision=2 * weight_decay,
    )
    la.fit(train_loader)

    if optimize_prior:
        la.optimize_prior_precision(method="marglik")

    print("Prior precision we are using")
    print(la.prior_precision)

    # and get some samples from it, our initial velocities
    # now I can get some samples for the Laplace approx
    if subset_of_weights == "last_layer":
        if hessian_structure == "diag":
            n_last_layer_weights = n_last_layer_weights
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().numpy()
        else:
            n_last_layer_weights = n_last_layer_weights
            dist = MultivariateNormal(loc=torch.zeros(n_last_layer_weights), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,))
            V_LA = V_LA.detach().numpy()
    else:
        if hessian_structure == "diag":
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            samples = torch.randn(n_posterior_samples, la.n_params, device=device).detach()
            samples = samples * la.posterior_scale.reshape(1, la.n_params)
            V_LA = samples.detach().numpy()

        else:
            dist = MultivariateNormal(loc=torch.zeros_like(map_solution), scale_tril=la.posterior_scale)
            V_LA = dist.sample((n_posterior_samples,))
            V_LA = V_LA.detach().numpy()
            print(V_LA.shape)

    # ok now I have the initial velocities. I can therefore consider my manifold

    if subset_of_weights == "last_layer":
        weights_ours = torch.zeros(n_posterior_samples, len(map_solution))
        # weights_LA = torch.zeros(n_posterior_samples, len(w_MAP))

        MAP = map_solution.clone()
        feature_extractor_map = MAP[0:-n_last_layer_weights]
        ll_map = MAP[-n_last_layer_weights:]
        print(feature_extractor_map.shape)
        print(ll_map.shape)

        # and now I have to define again the model
        feature_extractor_model = deepcopy(initial_model)
        ll = deepcopy(initial_model)[-1] # nn.Linear(H, num_output)

        # and use the correct weights
        torch.nn.utils.vector_to_parameters(feature_extractor_map, feature_extractor_model.parameters())
        torch.nn.utils.vector_to_parameters(ll_map, ll.parameters())

        # I have to precompute some stuff
        # i.e. I am treating the hidden activation before the last layer as my input
        # because since the weights are fixed, then this feature vector is fixed
        with torch.no_grad():
            R = feature_extractor_model(x_train)

        if optimize_prior:
            manifold = cross_entropy_manifold(
                ll, R, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2
            )

        else:
            manifold = cross_entropy_manifold(ll, R, y_train, batching=False, lambda_reg=weight_decay)

    else:
        model2 = deepcopy(initial_model)
        # here depending if I am using a diagonal approx, I have to redefine the model
        if optimize_prior:
            if batch_data:
                manifold = cross_entropy_manifold(
                    model2, train_loader, y=None, batching=True, lambda_reg=la.prior_precision.item() / 2
                )

            else:
                manifold = cross_entropy_manifold(
                    model2, x_train, y_train, batching=False, lambda_reg=la.prior_precision.item() / 2
                )
        else:
            if batch_data:
                manifold = cross_entropy_manifold(
                    model2, train_loader, y=None, batching=True, lambda_reg=weight_decay
                )

            else:
                manifold = cross_entropy_manifold(
                    model2, x_train, y_train, batching=False, lambda_reg=weight_decay
                )

    # now i have my manifold and so I can solve the expmap
    weights_ours = torch.zeros(n_posterior_samples, len(map_solution))
    for n in tqdm(range(n_posterior_samples), desc="Solving expmap"):
        v = V_LA[n, :].reshape(-1, 1)

        if subset_of_weights == "last_layer":
            curve, failed = geometry.expmap(manifold, ll_map.clone(), v)
            _new_ll_weights = curve(1)[0]
            _new_weights = torch.cat(
                (feature_extractor_map.view(-1), torch.from_numpy(_new_ll_weights).float().view(-1)), dim=0
            )
            weights_ours[n, :] = _new_weights.view(-1)
            torch.nn.utils.vector_to_parameters(_new_weights, model.parameters())

        else:
            # here I can try to sample a subset of datapoints, create a new manifold and solve expmap  
            curve, failed = geometry.expmap(manifold, map_solution.clone(), v)
            _new_weights = curve(1)[0]
            weights_ours[n, :] = torch.from_numpy(_new_weights.reshape(-1))

    # I can get the LA weights
    weights_LA = torch.zeros(n_posterior_samples, len(map_solution))

    for n in range(n_posterior_samples):
        if subset_of_weights == "last_layer":
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float() + ll_map.clone()
            laplace_weigths = torch.cat((feature_extractor_map.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[n, :] = laplace_weigths.cpu()
        else:
            laplace_weigths = torch.from_numpy(V_LA[n, :].reshape(-1)).float() + map_solution
            # laplace_weigths = torch.cat((feature_extractor_MAP.clone().view(-1), laplace_weigths.view(-1)), dim=0)
            weights_LA[n, :] = laplace_weigths.cpu()

    # now I can use my weights for prediction. Deoending if I am using linearization or not the prediction looks differently

    # N_grid = 100
    # offset = 2
    # x1min = x_train[:, 0].min() - offset
    # x1max = x_train[:, 0].max() + offset
    # x2min = x_train[:, 1].min() - offset
    # x2max = x_train[:, 1].max() + offset
    # x_grid = torch.linspace(x1min, x1max, N_grid)
    # y_grid = torch.linspace(x2min, x2max, N_grid)
    # XX1, XX2 = np.meshgrid(x_grid, y_grid)
    # X_grid = np.column_stack((XX1.ravel(), XX2.ravel()))

    # P_grid_LAPLACE = 0
    # for n in tqdm(range(n_posterior_samples), desc="computing laplace samples"):
    #     # put the weights in the model
    #     torch.nn.utils.vector_to_parameters(weights_LA[n, :], model.parameters())
    #     # compute the predictions
    #     with torch.no_grad():
    #         P_grid_LAPLACE += torch.softmax(model(torch.from_numpy(X_grid).float()), dim=1).numpy()

    # P_test_OURS = 0
    # for n in tqdm(range(n_posterior_samples), desc="computing laplace prediction in region"):
    #     # put the weights in the model
    #     torch.nn.utils.vector_to_parameters(weights_ours[n, :], model.parameters())
    #     # compute the predictions
    #     with torch.no_grad():
    #         P_test_OURS += torch.softmax(model(x_test), dim=1)


    # P_test_OURS /= n_posterior_samples
    # P_test_LAPLACE /= n_posterior_samples

    # accuracy_OURS = accuracy(P_test_OURS, y_test)

    # nll_OUR = nll(P_test_OURS, y_test)

    # brier_OURS = brier(P_test_OURS, y_test)

    # ece_our = calibration_error(P_test_OURS, y_test, norm="l1", task="multiclass", num_classes=2, n_bins=10) * 100
    # mce_our = calibration_error(P_test_OURS, y_test, norm="max", task="multiclass", num_classes=2, n_bins=10) * 100


    return weights_ours, weights_LA
