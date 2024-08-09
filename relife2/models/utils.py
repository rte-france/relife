# def regression_fitsearch(
#     regression: Union[ProportionalHazard, AFT],
#     time: ArrayLike,
#     event: Optional[ArrayLike] = None,
#     entry: Optional[ArrayLike] = None,
#     departure: Optional[ArrayLike] = None,
#     covar_list=list[FloatArray],
# ) -> list[FloatArray]:
#     """
#     Args:
#         regression ():
#         time ():
#         event ():
#         entry ():
#         departure ():
#         covar_list ():
#
#     Returns:
#         list of optmized params and fitting results per covar sample
#     """
#     for covar in covar_list:
#         if covar.shape[-1] != regression.functions.covar_effect.nb_params:
#             optimized_functions = type(regression.functions)(
#                 CovarEffect(**{f"coef_{i}": None for i in range(covar.shape[-1])}),
#                 regression.functions.baseline.copy(),
#             )
#         else:
#             optimized_functions = functions.copy()
