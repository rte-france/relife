.. ReLife2 documentation master file, created by
   sphinx-quickstart on Wed Feb 28 11:31:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ReLife2's documentation!
===================================

**Date**: |today| **Version**: |version|

ReLife is an open source Python library for asset management based on reliability theory and lifetime data analysis.

* **Survival analysis**: non-parametric estimator (Kaplan-Meier), parametric estimator (Maximum Likelihood) and regression models (Accelerated Failure Time and Parametric Proportional Hazards) on left-truncated, right-censored and left-censored lifetime data.
* **Reliability theory**: optimal age of replacement for time-based mainteance policy for one-cycle or infinite number of cycles, with exponential discounting.
* **Renewal theory**: expected number of events, expected total costs or expected number of replacements for run-to-failures or age replacement policies.

To install ReLife2, use pip:

.. code-block:: console

   pip install .


.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
         :class-card: intro-card

         New to ReLife? Check out the quick start page. It contains examples of main 
         ReLife commands and links to additional documentation.


         +++

         .. button-ref:: quick_start
            :ref-type: any
            :click-parent:
            :color: secondary
            :expand:

            Quick start

    .. grid-item-card::  User guide
         :class-card: intro-card
         :shadow: md

         The user guide provides in-depth information on the
         key concepts of ReLife with useful background information and explanation.

         +++

         .. button-ref:: user_guide/index
            :ref-type: any
            :click-parent:
            :color: secondary
            :expand:

            To the user guide

    .. grid-item-card::  API reference
         :class-card: intro-card

         The reference guide contains a detailed description of
         the ReLife API. The reference describes how the methods work and which parameters can
         be used. It assumes that you have an understanding of the key concepts and Python
         knowledge

         +++

         .. button-ref:: reference/index
            :ref-type: any
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide

    .. grid-item-card::  Contributor's guide
         :class-card: intro-card

         Saw a typo in the documentation? Want to improve
         existing functionalities? The contributing guidelines will guide
         you through the process of improving ReLife.

         +++

         .. button-ref:: contributor_guide/index
            :ref-type: any
            :click-parent:
            :color: secondary
            :expand:

            To the contributor's guide


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:


    quick_start
    user_guide/index
    reference/index
    contributor_guide/index







