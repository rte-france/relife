How to custom your databook
===========================

.. role:: python(code)
   :language: python

The databook function instanciates a :python:`DataBook` object. This functions take
several arguments (see :doc:`../detailed_doc/data`) but it also allows extra 
key-words arguments like :

* :python:`observed` : :python:`Data` instance (default to :python:`None`)
* :python:`left_censored` : :python:`Data` instance (default to :python:`None`)
* :python:`right_censored` : :python:`Data` instance (default to :python:`None`)
* :python:`interval_censored` : :python:`IntervalData` instance (default to :python:`None`)
* :python:`left_truncated` : :python:`Data` instance (default to :python:`None`)
* :python:`right_truncated` : :python:`Data` instance (default to :python:`None`)
* :python:`interval_truncated` : :python:`IntervalData` instance (default to :python:`None`)

Thus, if one wants to use its own data format, it is possible by specifying one or all
of the above arguments.

.. :warning::
    To do, object passed as arguments must be a :python:`Data` or :python:`IntervalData`
    instance

:python:`Data` or :python:`IntervalData` objects share a similar interface with two attributes

*  :python:`values` (:python:`np.ndarray`) : values of lifetimes, shape is always :python:`(n,)` in :python:`Data` and always :python:`(n,2)` in :python:`IntervalData`
*  :python:`index` (:python:`np.ndarray`) : index of lifetimes, shape is always :python:`(n,)`

They are initialized thanks to a method called :python:`parse` that must always return a tuple :python:`(index, values)`.
Then, if a user wants to implement its own :python:`Data` (or :python:`IntervalData` object), he would
write its own class like this : 

.. code-block:: python

    from relife2.data.object import Data

    class MyCustomData(Data)
        def __init__(self, *data):
            super().__init__(*data)

        def parse(self, *data):
            # a personal parsing process
            return index, values

After that, he could pass this new :python:`Data` in the desired extra key-word argument of :python:`databook`