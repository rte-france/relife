How to use databook
====================

.. role:: python(code)
   :language: python


.. image:: ../img/main_process_bis_1.png
    :scale: 100 %
    :align: center


One main object of ReLife is the ``DataBook``. It is used everywhere. As a book, it holds
every lifetime data information provided and allows users to explore data in a convenient
way : get all complete lifetime values, get right censored lifetime values being left truncated, etc.
Following previous section example, a databook is created as follow:

Instanciate a databook
----------------------

.. code-block:: python

    from relife2.data import databook

    first_db = databook(
        observed_lifetimes = first_data["observed_lifetimes"],
        right_censored_indicators = first_data["event"] == 0,
        complete_indicators = first_data["event"] == 1,
        entry = first_data["entry"],
    )

    second_db = databook(
        observed_lifetimes = second_data["observed_lifetimes"],
        entry = second_data["entry"],
        departure = second_data["departure"],
    )

As mentionned before, with 1d-array lifetimes, censored lifetimes must be explicitly
tagged through indicators. Here :python:`event` can serve both :python:`right_censored_indicators`
and :python:`complete_indicators`.

Databook exploration
--------------------

Now, lifetimes data can be explored very easily. For instance, one might want to get every
complete lifetimes. To do so just call:

>>> first_db("complete").values
np.array([10, 9, 11])

>>> second_db("complete").values
np.array([5, 10])

One can also get corresponding data index. Just replace :python:`.values` by :python:`.index`.

Databook can do more. One might wants to access lifetimes being complete **and** left truncated.
To do so, one can use the "and" operator as follow : 

>>> first_db("complete & left_truncated")[0].values
np.array([9, 11])

>>> first_db("complete & left_truncated")[1].values
np.array([3, 9])

The "or" operator can also be used. For instance :

>>> first_db("complete | left_truncated")[0].values
np.array([10, 9, 11])

Finally, a convenient method of databook is :python:`info`. It summarizes all the databook
content in one view :

>>> first_db.info()
     Lifetime data            Counts
 Nb samples (tot.)                12
          Observed                 3
     Left censored                 0
    Right censored                 4
 Interval censored                 0
    Left truncated                 5
   Right truncated                 0
Interval truncated                 0

>>> second_db.info()
     Lifetime data            Counts
 Nb samples (tot.)                13
          Observed                 2
     Left censored                 1
    Right censored                 1
 Interval censored                 3
    Left truncated                 2
   Right truncated                 1
Interval truncated                 3
