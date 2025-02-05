{% set survival_functions = ["chf", "hf", "dhf", "ichf", "isf", "jac_chf", "jac_hf", "mrl", "sf"] %}
{% set statistics = ["mean", "var", "moment", "median"] %}
{% set probability_functions = ["pdf", "ppf", "cdf"] %}
{% set exclude_methods = ["__init__", "__new__", "compose_with", "new_params", "init_params"] %}
{% set exclude_members = ", ".join(exclude_methods)  %}

{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :inherited-members:
    :exclude-members: {{ exclude_members }}

    .. rubric:: {{ _('Methods') }}

    **{{ _('Survival functions') }}**

    .. autosummary::
        :nosignatures:

    {% for item in methods %}
        {% if item in survival_functions %}
            ~{{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    .. autosummary::
        :nosignatures:

    {% for item in methods %}
        {% if item in probability_functions %}
            ~{{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    **{{ _('Statistics') }}**

    .. autosummary::
        :nosignatures:

    {% for item in methods %}
        {% if item in statistics %}
            ~{{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    **{{ _('Other methods') }}**

    .. autosummary::
        :nosignatures:

    {% for item in methods %}
        {% if (item not in survival_functions + statistics + probability_functions + exclude_methods) and not item.startswith("_") %}
            ~{{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}
