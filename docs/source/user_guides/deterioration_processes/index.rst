Deterioration processes
========================

Deterioration processes cover assets whose condition degrades gradually, so that the event of
interest is a threshold crossing or an imperfect repair rather than a plain failure:

- **gamma processes** (not implemented yet), for deterioration measured on the asset
  (corrosion, loss of thickness);
- **Kijima processes** (:py:class:`~relife.stochastic_processes.Kijima1Process`,
  :py:class:`~relife.stochastic_processes.Kijima2Process`), where a repair removes part of the
  accumulated damage, controlled by a rejuvenation parameter.

No maintenance policy is built on these processes yet.

.. toctree::
    :maxdepth: 1

    datasets
    gamma_process
    kijima_processes
    sampling
