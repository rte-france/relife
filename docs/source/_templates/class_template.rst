{# see https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html for more information on jinja2 templates #}
{% set exclude_methods = ["__init__", "__new__"] %}
{% set exclude_members = ", ".join(exclude_methods)  %}

{{ name | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :inherited-members:
    :exclude-members: {{ exclude_members }}
    :member-order: alphabetical

    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :nosignatures:

    {% for item in methods %}
        {% if item not in exclude_methods and not item.startswith('_') %}
            ~{{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}
    {% endif %}
