# TODO : deplacer tout ce module dans lifetime_model._base


# TODO : mettre dans lifetime_model._base (circular import)


# TODO : mettre dans lifetime_model._base (circular import)


# def _hessian_scheme(
#     likelihood: DefaultLifetimeLikelihood[M],
#     params: NDArray[np.float64],
#     method: Literal["2point", "cs"] = "cs",
#     eps: float = 1e-6,
# ) -> NDArray[np.float64]:
#     size = params.size
#     hess = np.empty((size, size))

#     # hessian 2 point
#     if method == "2point":
#         for i in range(size):
#             hess[i] = approx_fprime(
#                 params,
#                 lambda x: likelihood.jac_negative_log(x)[i],
#                 eps,
#             )
#         return hess
#     # hessian cs
#     u = eps * 1j * np.eye(size)
#     complex_params = params.astype(np.complex64)  # change params to complex
#     for i in range(size):
#         for j in range(i, size):
#             hess[i, j] = (
#                 np.imag(likelihood.jac_negative_log(complex_params + u[i])[j]) / eps
#             )
#             if i != j:
#                 hess[j, i] = hess[i, j]
#     return hess


# add approx_hessian str arg to options of fit instead of testing instance
# def approx_hessian(
#     likelihood: DefaultLifetimeLikelihood[M],
#     params: NDArray[np.float64],
#     eps: float = 1e-6,
# ) -> NDArray[np.float64]:
#     from relife.lifetime_model import Gamma
#     from relife.lifetime_model._parametric import ParametricLifetimeRegression

#     if isinstance(likelihood.model, ParametricLifetimeRegression):
#         if isinstance(likelihood.model.baseline, Gamma):
#             return _hessian_scheme(likelihood, params, method="2point", eps=eps)
#     if isinstance(likelihood.model, Gamma):
#         return _hessian_scheme(likelihood, params, method="2point", eps=eps)
#     return _hessian_scheme(likelihood, params, eps=eps)
