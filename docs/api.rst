API
===

Import cnvturbo together with scanpy as

.. code-block:: python

   import scanpy as sc
   import cnvturbo as cnv

For consistency, the cnvturbo API tries to follow the
`scanpy API <https://scanpy.readthedocs.io/en/stable/api/index.html>`__
as closely as possible.

The public surface area is intentionally small and focused on the
R-compatible ``inferCNV`` workflow. See :doc:`infercnv` for the full
pipeline overview and :doc:`dev_notes` for non-obvious lessons.

.. _api-io:

Input/Output: ``io``
--------------------

.. module:: cnvturbo.io

.. autosummary::
   :toctree: ./generated

   genomic_position_from_gtf
   genomic_position_from_biomart


Tools: ``tl``
-------------

.. module:: cnvturbo.tl

R-compatible inferCNV pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These three functions are the canonical workflow. Always call them in
this order on the same ``AnnData`` object.

.. autosummary::
   :toctree: ./generated

   infercnv_r_compat
   compute_hspike_emission_params
   hmm_call_subclusters

For per-cell HMM calls without leiden subclustering (rare; mostly used
for benchmarking the cluster-level shortcut):

.. autosummary::
   :toctree: ./generated

   hmm_call_cells

Per-cell scoring helpers
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ./generated

   cnv_score_cell

.. note::

   ``cnv_score_cell`` has fallback paths that may not align with R's
   ``cnv_signal_R``. For an R-equivalent per-cell signal, compute
   ``mean(|X_cnv_r - 1|)`` directly on ``adata.obsm["X_cnv_r"]``.
   See :doc:`dev_notes` § 2.


Plotting: ``pl``
----------------

.. module:: cnvturbo.pl

.. autosummary::
   :toctree: ./generated

   chromosome_heatmap
